import datetime
import os
from dateutil import tz
from argparse import Namespace, ArgumentParser
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from biovlp.models.cvt2distilgpt2 import Cvt2DistilGPT2Module
from biovlp.models.temporal_model import TemporalReportGenerationModule
from biovlp.models.clgen_model import CLGenerationModule
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
        ModelCheckpoint(monitor="val_chen_cider", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_chen_cider", min_delta=1e-4,
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
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/report_generation/{extension}/exp_logs")
    # initialize tokenizer
    # if hparams.model_name == "cvt2distilgpt2":
    #     model = Cvt2DistilGPT2Module(**vars(hparams))
    # elif hparams.model_name == "temporal_decoder":
    #     model = TemporalReportGenerationModule(**vars(hparams))
    #     # model = TemporalReportGenerationModule.load_from_checkpoint(
    #     #     hparams.ckpt_path, batch_size=hparams.batch_size)
    # elif hparams.model_name == "clgen":
    #     model = CLGenerationModule(**vars(hparams))
    # else:
    #     raise NotImplementedError
    # datamodule = model.datamodule

    # if hparams.ckpt_path:
    #     ckpt = torch.load(hparams.ckpt_path)
    #     msg = model.load_state_dict(ckpt["state_dict"], strict=False)
    #     print(msg)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    # trainer.fit(model, datamodule=datamodule)
    # # ckpt_path="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/temporal_decoder_2023_10_12_14_46_09/ckpts/epoch=7-step=9776.ckpt")
    # trainer.test(model, datamodule=datamodule, ckpt_path="best")
    # model = Cvt2DistilGPT2Module.load_from_checkpoint(
    #     "/home/fywang/Documents/Multi-seq-mae/data/backup_report_generation/cvt2distilgpt2_2023_11_02_15_41_16/ckpts/epoch=15-step=36368.ckpt",
    #     **vars(hparams))
    model = TemporalReportGenerationModule.load_from_checkpoint(
        "/home/fywang/Documents/Multi-seq-mae/data/backup_report_generation/temporal_decoder_2023_10_25_21_08_26/ckpts/last.ckpt",
        **vars(hparams))
    datamodule = model.datamodule
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    '''
    Command to run this script:
    CUDA_VISIBLE_DEVICES=0,1 python biovlp/tools/train_report_generation.py --model_name clgen --batch_size 16 --num_devices 2
    CUDA_VISIBLE_DEVICES=2,3 python biovlp/tools/train_report_generation.py --num_devices 2 --model_name cvt2distilgpt2 --batch_size 16
    CUDA_VISIBLE_DEVICES=1 python biovlp/tools/train_report_generation.py --model_name temporal_decoder --batch_size 4 --num_devices 1
    '''
    parser = ArgumentParser(description="Run model for report generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="cvt2distilgpt2",
                        choices=["cvt2distilgpt2", "temporal_decoder", "clgen"])
    parser.add_argument("--dataset_name", type=str, default="mimic_cxr",
                        choices=["mimic_cxr", "iu_xray"])
    parser.add_argument("--annotation_file", type=str,
                        default="/disk1/fywang/CXR_dataset/temporal_CXR/mimic_annotation.json"
                        )
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/fywang/CXR_dataset/mimic_data/2.0.0/files"
                        )
    parser.add_argument("--visual_model", type=str, default="microsoft/cvt-21-384-22k",
                        choices=["microsoft/cvt-21-384-22k", "resnet_50", "vit_base_patch16_384"])
    parser.add_argument("--num_devices", type=int, default=2)
    parser.add_argument("--ckpt_path", type=str,
                        default="")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/cvt2distilgpt2_2023_10_19_15_57_05/ckpts/epoch=13-step=31822.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/clgen_2023_10_23_14_59_17/ckpts/epoch=3-step=12124.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/cvt2distilgpt2_2023_10_27_11_29_19/ckpts/epoch=10-step=19206.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/cvt2distilgpt2_2023_10_29_11_40_07/ckpts/epoch=5-step=10476.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/clgen_2023_10_29_21_58_12/ckpts/epoch=3-step=6984.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/cvt2distilgpt2_2023_11_02_15_41_16/ckpts/epoch=15-step=36368.ckpt")
                        # default="/home/r15user2/Documents/Multi-seq-mae/data/report_generation/clgen_2023_11_03_13_26_45/ckpts/epoch=9-step=22730.ckpt")
    # default="")
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=5)
    hparams = parser.parse_args()

    seed_everything(hparams.seed)
    main(hparams)
