import datetime
import os
from typing import Any, Dict
from dateutil import tz
from argparse import Namespace, ArgumentParser
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import ipdb
from transformers import (AutoTokenizer, GPT2Config, GPT2TokenizerFast,
                          GPT2LMHeadModel, PretrainedConfig, EncoderDecoderModel)
from lightning import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from biovlp.datasets.mimic_cxr_dataset import MIMICLMDataset, custom_lm_collate_fn
from biovlp.datasets.datamodule import DataModule
from biovlp.backbones.lr_scheduler import linear_warmup_decay


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


class FineTuneGPT2Module(LightningModule):
    def __init__(self,
                 annotation_file: str,
                 dataset_dir: str,
                 language_model: str = "distilgpt2",
                 train_data_pct: float = 1.,
                 warmup_epochs: int = 10,
                 max_epochs: int = 50,
                 max_length: int = 128,
                 batch_size: int = 16,
                 image_size: int = 384,
                 num_workers: int = 16,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.05,
                 num_devices: int = 1,
                 accumulate_grad_batches: int = 4,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.language_model = language_model
        self.annotation_file = annotation_file
        self.dataset_dir = dataset_dir
        self.train_data_pct = train_data_pct
        self.max_length = max_length
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_devices = num_devices
        self.accumulate_grad_batches = accumulate_grad_batches

        self.tokenizer = self.setup_tokenizer()
        self.datamodule = self.setup_datamodule()

        # define distilGPT2 decoder
        config = GPT2Config.from_pretrained(self.language_model)
        # config.add_cross_attention = True
        config.is_decoder = True
        self.decoder = GPT2LMHeadModel.from_pretrained(
            self.language_model, config=config)
        # Resize GPT2 embedding to include padding and beginning of sentence token:
        self.decoder.resize_token_embeddings(config.vocab_size + 2)

    def forward(self, batch: Dict) -> Any:
        outputs = self.decoder(
            input_ids=batch["decoder_input_ids"],
            attention_mask=batch["decoder_attention_mask"],
            labels=batch["label_ids"])
        loss = outputs.loss
        return loss

    def training_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        batch_size = batch["decoder_input_ids"].shape[0]
        train_loss = self(batch)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        return train_loss

    def validation_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT | None:
        batch_size = batch["decoder_input_ids"].shape[0]
        val_loss = self(batch)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        return val_loss

    def test_step(self, batch: Dict, batch_idx: int):
        batch_size = batch["decoder_input_ids"].shape[0]
        test_loss = self(batch)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True,
                 prog_bar=True, batch_size=batch_size)
        return test_loss

    def setup_tokenizer(self):
        # Decoder tokenizer:
        tokenizer = GPT2TokenizerFast.from_pretrained(self.language_model)
        tokenizer.add_special_tokens(
            {"bos_token": "[BOS]", 'pad_token': '[PAD]'})

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(tokenizer, k + "_id")}')
            else:
                for i, j in zip(tokenizer.additional_special_tokens, tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')
        return tokenizer

    def setup_datamodule(self):
        '''
        define the datamodule for the model
        '''
        datamodule = DataModule(
            dataset=MIMICLMDataset,
            tokenizer=self.tokenizer,
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            max_length=self.max_length,
            train_data_pct=self.train_data_pct,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            collate_fn=custom_lm_collate_fn
        )

        self.train_iters_per_epoch = len(
            datamodule.train_dataloader()) // (self.num_devices * self.accumulate_grad_batches)

        return datamodule

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


'''
CUDA_VISIBLE_DEVICEs=1 python finetune_gpt2.py --num_devices 1
'''


def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"LM_{extension}"
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"data/report_generation/{extension}/ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=True, mode="min", save_top_k=1),
        EarlyStopping(monitor="val_loss", min_delta=1e-4,
                      patience=10, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(REPO_ROOT_DIR, "data/report_generation/logs")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="report_generation", save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        # gradient_clip_val=0.1,
        deterministic=True,
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
    model = FineTuneGPT2Module(**vars(hparams))
    datamodule = model.datamodule

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    parser = ArgumentParser(description="Run model for report generation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--annotation_file", type=str,
                        default="/home/r15user2/Documents/CXR_dataset/temporal_CXR/mimic_annotation.json"
                        )
    parser.add_argument("--dataset_dir", type=str,
                        default="/home/r15user2/Documents/CXR_dataset/mimic_data/2.0.0/files"
                        )
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    hparams = parser.parse_args()

    seed_everything(hparams.seed)
    main(hparams)
