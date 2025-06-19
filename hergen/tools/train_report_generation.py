import datetime
import os
from dateutil import tz
from argparse import Namespace, ArgumentParser
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from hergen.models.cvt2distilgpt2 import Cvt2DistilGPT2Module
from hergen.models.temporal_model import TemporalReportGenerationModule
from hergen.models.clgen_model import CLGenerationModule
import ipdb


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"{hparams.model_name}_{extension}"
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"data/report_generation/{extension}/ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_ce_f1_example", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_ce_f1_example", min_delta=0,
                      patience=10, verbose=False, mode="max")
    ]
    logger_dir = os.path.join(REPO_ROOT_DIR, "data/report_generation/logs")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="report_generation", save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        gradient_clip_val=0.1,
        # deterministic=True,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision="16",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/report_generation/{extension}/exp_logs")
    # initialize tokenizer
    if hparams.model_name == "cvt2distilgpt2":
        model = Cvt2DistilGPT2Module(**vars(hparams))
    elif hparams.model_name == "temporal_decoder":
        model = TemporalReportGenerationModule(**vars(hparams))
    elif hparams.model_name == "clgen":
        model = CLGenerationModule(**vars(hparams))
    else:
        raise NotImplementedError
    datamodule = model.datamodule

    if hparams.ckpt_path:
        ckpt = torch.load(hparams.ckpt_path)
        msg = model.load_state_dict(ckpt["state_dict"], strict=False)
        print(msg)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    # trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    parser = ArgumentParser(description="Run model for report generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="cvt2distilgpt2",
                        choices=["cvt2distilgpt2", "temporal_decoder", "clgen"])
    parser.add_argument("--dataset_name", type=str, default="mimic_cxr",
                        choices=["mimic_cxr", "iu_xray"])
    parser.add_argument("--annotation_file", type=str,
                        default="/disk1/*/CXR_dataset/temporal_CXR/mimic_annotation.json"
                        )
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/*/CXR_dataset/mimic_data/2.0.0/files"
                        )
    parser.add_argument("--visual_model", type=str, default="microsoft/cvt-21-384-22k")
    parser.add_argument("--num_devices", type=int, default=2)
    parser.add_argument("--ckpt_path", type=str,
                        default="")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=5)
    parser.add_argument("--freeze_visual_model", action="store_true")
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=1e-5)
    hparams = parser.parse_args()

    seed_everything(hparams.seed)
    main(hparams)
