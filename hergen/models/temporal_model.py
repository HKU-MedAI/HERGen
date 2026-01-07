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
from hergen.datasets.temporal_mimic_cxr_dataset import TemporalMIMICCXRDataset, temporal_collate_fn
# from hergen.models.miniGPT import GPT
from hergen.models.group_causal_transformer import TemporalAggregator
from hergen.backbones.lr_scheduler import linear_warmup_decay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


class TemporalReportGenerationModule(Cvt2DistilGPT2Module):
    ''' Temporal report generation architecture '''

    def __init__(self,
                 dataset_name: str,
                 annotation_file: str,
                 dataset_dir: str,
                 exp_log_dir: str,
                 visual_model: str = "microsoft/cvt-21-384-22k",
                 language_model: str = "distilgpt2",
                 freeze_text_encoder: bool = True,
                 train_data_pct: float = 1.,
                 max_seq_len: int = 5,
                 num_heads: int = 8,
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

        if self.visual_model == "resnet_50":
            group_size = 49
        elif self.visual_model == "vit_base_patch16_384":
            group_size = 96
        elif self.visual_model == "microsoft/cvt-21-384-22k":
            group_size = 50
        else:
            raise NotImplementedError

        # define temporal aggregation module
        self.visual_temporal_aggregator = TemporalAggregator(
            group_size=group_size,
            block_size=max_seq_len,
            n_embd=768,
            num_heads=num_heads,
            embd_pdrop=0.1,
            n_layer=2
        )
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads

        self.img_emb_projection = nn.Linear(768, 128)

        # define CXR-BERT
        url = "microsoft/BiomedVLP-CXR-BERT-general"
        self.encode_tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True)
        self.text_encoder = AutoModel.from_pretrained(
            url, trust_remote_code=True)

        # just for debug
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.img_emb_projection.parameters():
            param.requires_grad = False

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder_projection.parameters():
            param.requires_grad = False

        for param in self.encoder_compact.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def encoder_forward(self, images, report, batch_mask, study_date):
        '''
        Forward step of image encoder
        '''

        batch_len = batch_mask.sum(dim=1)

        B, T, C, H, W = images.shape
        nonull_images = images[batch_mask]
        if "cvt" in self.visual_model:
            image_features = self.encoder(nonull_images)['last_hidden_state']
            image_features = image_features.permute(0, 2, 1)
            image_features = self.encoder_compact(image_features)
        elif self.visual_model == "vit_base_patch16_384":
            image_features = self.encoder(nonull_images, return_features=True)
            image_features = image_features[:, 1:]
        elif self.visual_model == "resnet_50":
            # setup forward for biovil-T
            _, image_features = self.encoder(nonull_images)
        else:
            raise NotImplementedError
        del images
        image_embs = self.encoder_projection(image_features)

        nonull_image_embs_list = torch.split(
            image_embs, batch_len.tolist(), dim=0)
        pad_image_embs = pad_sequence(
            nonull_image_embs_list, batch_first=True)
        pad_img_len = self.max_seq_len - pad_image_embs.shape[1]
        B, _, N, D = pad_image_embs.shape
        zero_image_embs = torch.zeros(
            B, pad_img_len, N, D).type_as(image_embs)
        image_embs = torch.cat(
            [pad_image_embs, zero_image_embs], dim=1)

        temporal_image_embs = self.visual_temporal_aggregator(
            image_embs, study_date)
        temporal_image_embs = rearrange(
            temporal_image_embs, "b (t n) d -> b t n d", b=B, t=T)
        temporal_img_embs = temporal_image_embs[batch_mask]

        # Here we create contrastive loss for image and report
        img_embs = self.img_emb_projection(
            torch.mean(temporal_img_embs, dim=1))

        if report is not None:
            batch_report = [item for sublist in report
                            for item in sublist]

            tokenized_data = self.encode_tokenizer(
                batch_report, add_special_tokens=True,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length)
            input_ids = tokenized_data['input_ids'].type_as(
                temporal_image_embs).long()
            attention_mask = tokenized_data['attention_mask'].type_as(
                temporal_image_embs).long()
            text_embs = self.text_encoder.get_projected_text_embeddings(
                input_ids=input_ids, attention_mask=attention_mask)

            # TODO: soft contrastive loss
            cont_loss = self.infonce_loss(img_embs, text_embs, 0.07)
        else:
            cont_loss = torch.tensor(0.).type_as(temporal_img_embs)

        return temporal_image_embs, cont_loss

    @staticmethod
    def infonce_loss(out_1, out_2, softmax_temperature):
        batch_size = out_1.size(0)
        logits = out_1 @ out_2.t()
        target_idxs = torch.arange(batch_size).type_as(logits).long()
        loss0 = F.cross_entropy(logits / softmax_temperature, target_idxs)
        loss1 = F.cross_entropy(logits.t() / softmax_temperature, target_idxs)
        cont_loss = (loss0 + loss1) / 2.

        return cont_loss

    def forward(self, images, input_ids, attention_mask, batch_mask, report, study_date):

        temporal_img_embs, cont_loss = self.encoder_forward(
            images, report, batch_mask, study_date)

        temporal_nonull_img_embs = temporal_img_embs[batch_mask]
        encoder_outputs = BaseModelOutput(
            last_hidden_state=temporal_nonull_img_embs)

        batch_input_ids = rearrange(input_ids, "b t d -> (b t) d")
        batch_attention_mask = rearrange(attention_mask, "b t d -> (b t) d")

        # only compute loss on meaning reports
        batch_input_ids = batch_input_ids[batch_mask.reshape(-1)]
        batch_attention_mask = batch_attention_mask[batch_mask.reshape(-1)]

        # Teacher forcing: labels are given as input
        outputs = self.decoder.encoder_decoder(
            decoder_input_ids=batch_input_ids,
            decoder_attention_mask=batch_attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits, cont_loss

    def generate(self, num_beams, batch_mask, images, study_date, report=None):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for the encoder.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        # FIXME: This part code is used for recursive generation, but we have a more efficient way now.
        # output_ids = []
        # batch_size = images.size(0)
        # max_seq_len = images.size(1)
        # # recursively generate report
        # for i in range(max_seq_len):
        #     cur_images = images.clone()
        #     cur_images[:, i+1:] = 0
        #     cur_study_date = study_date.clone()
        #     cur_study_date[:, i+1:] = 0
        #     cur_batch_mask = batch_mask.clone()
        #     cur_batch_mask[:, i+1:] = False
        #     temporal_img_embs, _ = self.encoder_forward(
        #         cur_images, report, cur_batch_mask, cur_study_date)

        #     # only use the image embeddings in the last timestamp
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=temporal_img_embs[:, i])

        #     outputs = self.decoder.encoder_decoder.generate(
        #         max_length=self.max_length,
        #         bos_token_id=self.tokenizer.bos_token_id,
        #         eos_token_id=self.tokenizer.eos_token_id,
        #         pad_token_id=self.tokenizer.pad_token_id,
        #         num_beams=num_beams,
        #         return_dict_in_generate=True,
        #         use_cache=True,
        #         encoder_outputs=encoder_outputs,
        #     )
        #     # make sure each decoded report is 128 tokens
        #     pad_length = self.max_length - outputs['sequences'].shape[1]
        #     if pad_length > 0:
        #         pad_sequence = self.tokenizer.pad_token_id * \
        #             torch.ones(batch_size, pad_length).type_as(images).long()
        #         output_sequence = torch.cat(
        #             (outputs["sequences"], pad_sequence), dim=1)
        #     else:
        #         output_sequence = outputs["sequences"]
        #     output_ids.append(output_sequence.unsqueeze(1))

        # output_ids = torch.cat(output_ids, dim=1)
        # output_nonull_ids = output_ids[batch_mask]
        # return output_nonull_ids

        # TODO: note that this is equivalent to recursively generation
        temporal_img_embs, _ = self.encoder_forward(
            images, report, batch_mask, study_date)

        temporal_nonull_img_embs = temporal_img_embs[batch_mask]
        # only use the image embeddings in the last timestamp
        encoder_outputs = BaseModelOutput(
            last_hidden_state=temporal_nonull_img_embs)

        outputs = self.decoder.encoder_decoder.generate(
            max_length=self.max_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    def training_step(self, batch: Dict, batch_idx: int):
        y_hat, cont_loss = self(
            batch["images"], batch["decoder_input_ids"], batch["decoder_attention_mask"],
            batch["batch_mask"], batch["report"], batch["study_date"]
        )

        batch_label_ids = rearrange(batch['label_ids'], "b t d -> (b t) d")
        batch_label_ids = batch_label_ids[batch["batch_mask"].reshape(-1)]
        # Loss:
        text_decode_loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]).contiguous(), batch_label_ids, ignore_index=self.tokenizer.pad_token_id,
        )
        loss = text_decode_loss + 2 * cont_loss

        log_dict = {
            'train_loss': loss,
            'train_text_decode_loss': text_decode_loss,
            'train_cont_loss': cont_loss
        }

        # Logging:
        self.log_dict(log_dict, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=y_hat.shape[0], sync_dist=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        output_ids = self.generate(
            self.num_beams, batch["batch_mask"],
            batch["images"], batch["study_date"], batch["report"])

        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        batch_ids = [item for sublist in batch['ids'] for item in sublist]
        batch_report = [item for sublist in batch['report']
                        for item in sublist]
        # Log reports:
        self.val_report_logger.update(generated, dicom_ids=batch_ids)

        # Evaluate:
        self.val_chexbert_metrics.update(
            generated, batch_report, ids=batch_ids)
        self.val_coco_metrics.update(
            generated, batch_report, ids=batch_ids)

        if batch_idx == 0:
            print("=" * 10)
            print("generted reports: ")
            print("\n".join(generated[:5]))
            print("="*10)
            print("ground truth reports: ")
            print("\n".join(batch_report[:5]))
            print("="*10)

    def test_step(self, batch, batch_idx):
        # Beam search:
        output_ids = self.generate(
            self.num_beams, batch["batch_mask"],
            batch['images'], batch["study_date"])

        # Generated report:
        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        batch_ids = [item for sublist in batch['ids'] for item in sublist]
        batch_report = [item for sublist in batch['report']
                        for item in sublist]

        # Log reports:
        self.test_report_logger.update(generated, dicom_ids=batch_ids)

        # Evaluate:
        self.test_chexbert_metrics.update(
            generated, batch_report, ids=batch_ids)
        self.test_coco_metrics.update(
            generated, batch_report, ids=batch_ids)

    def configure_optimizers(self) -> Any:
        # This optimizer actually helps
        # grouped_parameters = [
        #     {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
        #     {"params": self.visual_temporal_aggregator.parameters(),
        #      'lr': self.decoder_lr},
        #     {"params": self.encoder_compact.parameters(), 'lr': self.decoder_lr},
        #     {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
        #     {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
        # ]

        # optimizer = torch.optim.AdamW(
        #     grouped_parameters, lr=self.decoder_lr)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.encoder_lr)

        # optimizer = torch.optim.AdamW(
        #     self.visual_temporal_aggregator.parameters(), lr=self.decoder_lr)

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
    #     # This optimizer actually helps
    #     grouped_parameters = [
    #         # {"params": self.encoder.parameters(), 'lr': self.encoder_lr},
    #         {"params": self.visual_temporal_aggregator.parameters(),
    #          'lr': self.decoder_lr},
    #         # {"params": self.encoder_projection.parameters(), 'lr': self.decoder_lr},
    #         # {"params": self.encoder_compact.parameters(), 'lr': self.decoder_lr},
    #         # {"params": self.decoder.parameters(), 'lr': self.decoder_lr},
    #     ]

    #     optimiser = {'optimizer': torch.optim.AdamW(
    #         grouped_parameters, lr=self.decoder_lr)}
    #     return optimiser

    def setup_datamodule(self):

        self.datamodule = DataModule(
            dataset=TemporalMIMICCXRDataset,
            tokenizer=self.tokenizer,
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            max_length=self.max_length,
            train_data_pct=self.train_data_pct,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            image_size=self.image_size,
            collate_fn=temporal_collate_fn
        )

        # self.train_iters_per_epoch = 1000
        self.train_iters_per_epoch = len(
            self.datamodule.train_dataloader()) // (self.num_devices * self.accumulate_grad_batches)

        return self.datamodule
