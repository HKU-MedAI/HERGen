import datetime
import os
from copy import deepcopy as c
from dateutil import tz
from argparse import Namespace, ArgumentParser
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from biovlp.models.r2gen import R2Gen
# from merg.tools.path import DATAPATH


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
    extension = f'{hparams.model_name}_{extension}'
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"data/report_generation/{extension}/ckpts")
    # hparams.annotation_file = DATAPATH[hparams.dataset_name]['annotation_file']
    # hparams.dataset_dir = DATAPATH[hparams.dataset_name]['dataset_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_chen_bleu_4", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=1),
        EarlyStopping(monitor="val_chen_bleu_4", min_delta=0,
                      patience=5, verbose=True, mode="max")
    ]
    logger_dir = os.path.join(REPO_ROOT_DIR, "data/report_generation/logs")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="report_generation", save_dir=logger_dir, name=extension)

    if hparams.test_only:
        hparams.num_devices = 1

    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        accumulate_grad_batches=1,
        # gradient_clip_val=0.1,
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
    args = c(hparams)
    hparams.args = args
    if hparams.model_name == "r2gen":
        if args.ckpt_path:
            model = R2Gen.load_from_checkpoint(
                args.ckpt_path, **vars(hparams))
        else:
            model = R2Gen(**vars(hparams))
    else:
        raise NotImplementedError
    datamodule = model.datamodule
    # ------------------------
    # 3 START TRAINING
    # ------------------------
    if not hparams.test_only:
        trainer.fit(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule, ckpt_path='best')
    else:
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    '''
    Command to run this script:
    CUDA_VISIBLE_DEVICES=2,3 python biovlp/tools/train_r2gen.py --num_devices 2 --batch_size 16 --annotation_file /disk1/*/CXR_dataset/temporal_CXR/mimic_annotation.json
    CUDA_VISIBLE_DEVICES=0,1 python biovlp/tools/train_r2gen.py --num_devices 1 --batch_size 16 --test_only \
        --ckpt_path /home/*/Documents/Multi-seq-mae/data/report_generation/r2gen_2024_02_29_10_58_46/ckpts/last.ckpt
    '''
    parser = ArgumentParser(
        description="Run downstream task of few shot learning")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str,
                        default="")
    parser.add_argument("--random_state", type=int, default=42)  # 42
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_devices", type=int, default=1)
    # model

    parser.add_argument("--model_name", type=str, default="r2gen")
    parser.add_argument("--train_data_pct", type=float, default=1)

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='mimic_cxr',
                        choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/*/CXR_dataset/mimic_data/2.0.0/files")
    parser.add_argument("--annotation_file", type=str,
                        default="/disk1/*/CXR_dataset/knowledge_graph/mimic_annotation.json")
    parser.add_argument('--max_seq_length', type=int, default=60,
                        help='the maximum sequence length of the reports.')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str,
                        default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool,
                        default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512,
                        help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048,
                        help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0,
                        help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0,
                        help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0,
                        help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0,
                        help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                        help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3,
                        help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8,
                        help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int,
                        default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float,
                        default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1,
                        help='the sample number per image.')
    parser.add_argument('--group_size', type=int,
                        default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int,
                        default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int,
                        default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int,
                        default=1, help='whether to use block trigrams.')

    # Optimization
    parser.add_argument('--lr_ve', type=float, default=5e-5,
                        help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4,
                        help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR',
                        help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50,
                        help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the gamma of the learning rate scheduler.')

    hparams = parser.parse_args()
    print(hparams)

    seed_everything(hparams.random_state)
    main(hparams)
