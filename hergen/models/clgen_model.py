from typing import Any, Dict
from copy import deepcopy
import os
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from einops import rearrange
from transformers.modeling_outputs import BaseModelOutput
from hergen.models.cvt2distilgpt2 import Cvt2DistilGPT2Module
from hergen.datasets.datamodule import DataModule
# from hergen.datasets.temporal_mimic_cxr_dataset import TemporalMIMICCXRDataset, temporal_collate_fn
from hergen.datasets.mimic_cxr_dataset import MIMICCXRDataset
from hergen.datasets.base_dataset import custom_collate_fn
# from hergen.models.miniGPT import GPT
from hergen.models.group_causal_transformer import TemporalAggregator
from hergen.backbones.lr_scheduler import linear_warmup_decay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


class CLGenerationModule(Cvt2DistilGPT2Module):
    ''' Temporal report generation architecture '''

    def __init__(self,
                 dataset_name: str,
                 annotation_file: str,
                 dataset_dir: str,
                 exp_log_dir: str,
                 visual_model: str = "microsoft/cvt-21-384-22k",
                 language_model: str = "distilgpt2",
                 freeze_text_encoder: bool = False,
                 train_data_pct: float = 1.,
                 max_length: int = 128,
                 batch_size: int = 16,
                 image_size: int = 512,
                 num_workers: int = 16,
                 encoder_lr: float = 5e-5,
                 decoder_lr: float = 5e-4,
                 num_beams: int = 3,
                 gpt2_ckpt_path: str = "",
                 max_epochs: int = 50,
                 num_devices: int = 2,
                 accumulate_grad_batches: int = 1,
                 *args,
                 **kwargs) -> None:

        self.max_epochs = max_epochs
        self.warmup_epochs = int(0.2 * self.max_epochs)
        self.num_devices = num_devices
        self.accumulate_grad_batches = accumulate_grad_batches

        super().__init__(dataset_name, annotation_file, dataset_dir, exp_log_dir, visual_model, language_model,
                         train_data_pct, max_length, batch_size, image_size, num_workers, encoder_lr, decoder_lr,
                         num_beams, gpt2_ckpt_path)

        assert dataset_name == "mimic_cxr"

        self.img_emb_projection = nn.Linear(768, 128)

        # define CXR-BERT
        url = "microsoft/BiomedVLP-CXR-BERT-general"
        self.encode_tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(
            url, trust_remote_code=True)

        # freeze the CXR-BERT
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def encoder_forward(self, images, report=None):
        '''
        Forward step of image encoder
        '''

        if "cvt" in self.visual_model:
            image_features = self.encoder(images)['last_hidden_state']
            image_features = image_features.permute(0, 2, 1)
            image_features = self.encoder_compact(image_features)
        elif self.visual_model == "resnet_50":
            _, image_features = self.encoder(images)
        elif self.visual_model == "vit_base_patch16_384":
            image_features = self.encoder(images, return_features=True)
            image_features = image_features[:, 1:]
        else:
            raise NotImplementedError
        del images

        # compact the image features
        image_features = self.encoder_projection(image_features)
        image_features = image_features.clone()

        encoder_outputs = BaseModelOutput(last_hidden_state=image_features)

        if report is not None:
            tokenized_data = self.encode_tokenizer(
                report, add_special_tokens=True,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length)
            input_ids = tokenized_data['input_ids'].type_as(
                image_features).long()
            attention_mask = tokenized_data['attention_mask'].type_as(
                image_features).long()
            text_embs = self.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids, attention_mask=attention_mask)

            # TODO: soft contrastive loss
            image_embs = self.img_emb_projection(image_features)
            img_embs = torch.mean(image_embs, dim=1)

            cont_loss = self.infonce_loss(img_embs, text_embs, 0.07)

            return encoder_outputs, cont_loss

        else:
            return encoder_outputs

    @staticmethod
    def infonce_loss(out_1, out_2, softmax_temperature):
        batch_size = out_1.size(0)
        out_1 = F.normalize(out_1, dim=-1)
        out_2 = F.normalize(out_2, dim=-1)
        sim = out_2.detach() @ out_2.detach().t()
        lambda_ = 1.
        targets = lambda_ * \
            torch.eye(batch_size).type_as(sim) + (1 - lambda_) * sim

        logits = out_1 @ out_2.t()
        loss0 = F.cross_entropy(logits / softmax_temperature, targets)
        loss1 = F.cross_entropy(logits.t() / softmax_temperature, targets)
        cont_loss = (loss0 + loss1) / 2.

        return cont_loss

    def forward(self, images, input_ids, attention_mask, report):

        encoder_outputs, cont_loss = self.encoder_forward(
            images, report)

        # Teacher forcing: labels are given as input
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits, cont_loss

    def training_step(self, batch: Dict, batch_idx: int) -> STEP_OUTPUT:
        y_hat, cont_loss = self(
            batch["images"], batch["decoder_input_ids"], batch["decoder_attention_mask"], batch["report"]
        )

        # Loss:
        ce_loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]).contiguous(), batch['label_ids'], ignore_index=self.tokenizer.pad_token_id,
        )

        loss_dict = {
            "train_loss": ce_loss + 0.5 * cont_loss,
            "train_ce_loss": ce_loss,
            "train_cont_loss": cont_loss
        }

        # Logging:
        self.log_dict(loss_dict, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=y_hat.shape[0], sync_dist=True)

        return loss_dict["train_loss"]

    def configure_optimizers(self) -> Any:
        # This optimizer actually helps
        # grouped_parameters = [
        #     {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
        #     {"params": self.text_encoder.parameters(),
        #      'lr': self.decoder_lr},
        #     {"params": self.encoder_compact.parameters(), 'lr': self.decoder_lr},
        #     {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
        # ]

        optimizer = torch.optim.AdamW(
            self.parameters(),
            # lr=self.decoder_lr)
            lr=self.encoder_lr)

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

    # def configure_optimizers(self) -> Any:
    #     # # This optimizer actually helps
    #     grouped_parameters = [
    #         {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
    #         {"params": self.text_encoder.parameters(),
    #          'lr': self.decoder_lr},
    #         {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
    #         {"params": self.encoder_compact.parameters(), 'lr': self.decoder_lr},
    #         {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
    #     ]

    #     optimiser = {'optimizer': torch.optim.AdamW(
    #         grouped_parameters, lr=self.decoder_lr)}

    #     # optimiser = {'optimizer': torch.optim.AdamW(
    #     #     self.parameters(), lr=self.decoder_lr)}

    #     return optimiser

    def setup_datamodule(self):

        self.datamodule = DataModule(
            dataset=MIMICCXRDataset,
            tokenizer=self.tokenizer,
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            max_length=self.max_length,
            train_data_pct=self.train_data_pct,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            collate_fn=custom_collate_fn
        )

        self.train_iters_per_epoch = len(
            self.datamodule.train_dataloader()) // (self.num_devices * self.accumulate_grad_batches)

        return self.datamodule
