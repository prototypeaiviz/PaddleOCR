# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# add current folder and parent folder to python path so ppocr modules can be imported
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import yaml
import paddle
import paddle.distributed as dist  # used when training on multiple GPUs

# importing main PaddleOCR modules used to build different parts of the pipeline
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.losses import build_loss
from ppocr.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static

# program.py contains the training loop logic
import tools.program as program
import tools.naive_sync_bn as naive_sync_bn

dist.get_world_size()  # get number of distributed processes


def main(config, device, logger, vdl_writer, seed):

    # initialize distributed training if enabled
    if config["Global"]["distributed"]:
        dist.init_parallel_env()

    global_config = config["Global"]

    # build training dataloader (loads images + labels)
    set_signal_handlers()
    train_dataloader = build_dataloader(config, "Train", device, logger, seed)

    # check if dataset is empty
    if len(train_dataloader) == 0:
        logger.error(
            "No Images in train dataset, please ensure\n"
            + "\t1. dataset size >= batch size\n"
            + "\t2. label file path is correct."
        )
        return

    # build validation dataloader if evaluation is enabled
    if config["Eval"]:
        valid_dataloader = build_dataloader(config, "Eval", device, logger, seed)
    else:
        valid_dataloader = None

    step_pre_epoch = len(train_dataloader)  # number of batches per epoch

    # post process converts raw model output into text
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # if recognition model uses character dictionary
    if hasattr(post_process_class, "character"):

        char_num = len(getattr(post_process_class, "character"))  # number of characters

        # handle distillation models
        if config["Architecture"]["algorithm"] in ["Distillation"]:

            for key in config["Architecture"]["Models"]:

                # if multi-head architecture is used
                if config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead":

                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2

                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3

                    out_channels_list = {}
                    out_channels_list["CTCLabelDecode"] = char_num

                    # adjust SAR loss ignore index
                    if list(config["Loss"]["loss_config_list"][-1].keys())[0] == "DistillationSARLoss":

                        config["Loss"]["loss_config_list"][-1]["DistillationSARLoss"]["ignore_index"] = (char_num + 1)

                        out_channels_list["SARLabelDecode"] = char_num + 2

                    elif any("DistillationNRTRLoss" in d for d in config["Loss"]["loss_config_list"]):

                        out_channels_list["NRTRLabelDecode"] = char_num + 3

                    config["Architecture"]["Models"][key]["Head"]["out_channels_list"] = out_channels_list

                else:
                    config["Architecture"]["Models"][key]["Head"]["out_channels"] = char_num

        # multi-head recognition model
        elif config["Architecture"]["Head"]["name"] == "MultiHead":

            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2

            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3

            out_channels_list = {}
            out_channels_list["CTCLabelDecode"] = char_num

            if list(config["Loss"]["loss_config_list"][1].keys())[0] == "SARLoss":

                if config["Loss"]["loss_config_list"][1]["SARLoss"] is None:
                    config["Loss"]["loss_config_list"][1]["SARLoss"] = {"ignore_index": char_num + 1}
                else:
                    config["Loss"]["loss_config_list"][1]["SARLoss"]["ignore_index"] = (char_num + 1)

                out_channels_list["SARLabelDecode"] = char_num + 2

            elif list(config["Loss"]["loss_config_list"][1].keys())[0] == "NRTRLoss":

                out_channels_list["NRTRLabelDecode"] = char_num + 3

            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list

        else:
            # basic recognition model
            config["Architecture"]["Head"]["out_channels"] = char_num

        # SAR model special loss setting
        if config["PostProcess"]["name"] == "SARLabelDecode":
            config["Loss"]["ignore_index"] = char_num - 1

    # build the neural network architecture defined in YAML
    model = build_model(config["Architecture"])

    use_sync_bn = config["Global"].get("use_sync_bn", False)

    # convert batchnorm to sync batchnorm for multi GPU training
    if use_sync_bn:

        if config["Global"].get("use_npu", False) or config["Global"].get("use_xpu", False):
            naive_sync_bn.convert_syncbn(model)
        else:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        logger.info("convert_sync_batchnorm")

    # convert model to static graph for better performance
    model = apply_to_static(model, config, logger)

    # build loss function (how prediction error is calculated)
    loss_class = build_loss(config["Loss"])

    # build optimizer (updates model weights)
    optimizer, lr_scheduler = build_optimizer(
        config["Optimizer"],
        epochs=config["Global"]["epoch_num"],
        step_each_epoch=len(train_dataloader),
        model=model,
    )

    # build evaluation metric (accuracy etc.)
    eval_class = build_metric(config["Metric"])

    logger.info("train dataloader has {} iters".format(len(train_dataloader)))

    if valid_dataloader is not None:
        logger.info("valid dataloader has {} iters".format(len(valid_dataloader)))

    # AMP = mixed precision training (faster on GPU)
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_dtype = config["Global"].get("amp_dtype", "float16")

    if use_amp:
        scale_loss = config["Global"].get("scale_loss", 1.0)

        scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)

        if amp_level == "O2":
            model, optimizer = paddle.amp.decorate(
                models=model,
                optimizers=optimizer,
                level=amp_level,
                master_weight=True,
                dtype=amp_dtype,
            )
    else:
        scaler = None

    # load pretrained model if provided (used for fine tuning)
    pre_best_model_dict = load_model(
        config, model, optimizer, config["Architecture"]["model_type"]
    )

    # wrap model for distributed training
    if config["Global"]["distributed"]:
        find_unused_parameters = config["Global"].get("find_unused_parameters", False)

        model = paddle.DataParallel(
            model, find_unused_parameters=find_unused_parameters
        )

    # start the actual training process
    program.train(
        config,
        train_dataloader,
        valid_dataloader,
        device,
        model,
        loss_class,
        optimizer,
        lr_scheduler,
        post_process_class,
        eval_class,
        pre_best_model_dict,
        logger,
        step_pre_epoch,
        vdl_writer,
        scaler,
        amp_level,
        [],
        [],
        amp_dtype,
    )


# helper function to test dataloader speed
def test_reader(config, device, logger):

    loader = build_dataloader(config, "Train", device, logger)

    import time

    starttime = time.time()
    count = 0

    try:
        for data in loader():

            count += 1

            if count % 1 == 0:

                batch_time = time.time() - starttime
                starttime = time.time()

                logger.info(
                    "reader: {}, {}, {}".format(count, len(data[0]), batch_time)
                )

    except Exception as e:
        logger.info(e)

    logger.info("finish reader: {}, Success!".format(count))


if __name__ == "__main__":

    # load configuration and prepare environment
    config, device, logger, vdl_writer = program.preprocess(is_train=True)

    # set random seed for reproducibility
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)

    # run training
    main(config, device, logger, vdl_writer, seed)

    # test_reader(config, device, logger)