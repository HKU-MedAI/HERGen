import torch
import numpy as np
from typing import Any
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

from hergen.ext.r2gen import VisualExtractor, EncoderDecoder, compute_loss
from hergen.models.base_model import BaseLightningModule


class R2Gen(BaseLightningModule):
    def __init__(self,
                 args,
                 dataset_name: str,
                 annotation_file: str,
                 dataset_dir: str,
                 exp_log_dir: str = '',
                 visual_model: str = 'resnet101',      #
                 visual_model_pretrained: bool = True,
                 language_model: str = 'allenai/scibert_scivocab_uncased',
                 train_data_pct: float = 1.,
                 max_length: int = 128,
                 batch_size: int = 16,
                 image_size: int = 224,
                 num_beams: int = 3,
                 num_workers: int = 8,
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.05,
                 **kwargs,
                 ) -> None:
        """ Note that language_model is only used as a tokenizer;
            R2Gen has its own transformer architecture,
            bert here do not serve any function in loss and inference.
        """

        if args.visual_extractor == 'resnet101':
            self.image_size = 224
        else:
            raise NotImplementedError

        super().__init__(dataset_name, annotation_file, dataset_dir, exp_log_dir, language_model, train_data_pct,
                         max_length, num_beams, batch_size, image_size, num_workers)

        # setup args
        self.args = args
        # setup encoder
        self.encoder = VisualExtractor(args)
        # setup decoder
        self.encoder_decoder = EncoderDecoder(args, self.tokenizer)
        self.loss = compute_loss

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def setup_tokenizer(self):
        tokenizer = super().setup_tokenizer()
        return self.add_special_all_special_tokens(tokenizer)

    def encoder_forward(self, images):
        if self.dataset_name == "iu_xray":
            att_feats_0, fc_feats_0 = self.encoder(images[:, 0])
            att_feats_1, fc_feats_1 = self.encoder(images[:, 1])
            fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
            att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        elif self.dataset_name == "mimic_cxr":
            att_feats, fc_feats = self.encoder(images)
        else:
            raise NotImplementedError
        return att_feats, fc_feats

    def forward(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.encoder_forward(images)
        if mode == 'train':
            output = self.encoder_decoder(
                fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(
                fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        output = self(batch['images'],
                      batch['decoder_input_ids'], mode='train')
        loss = self.loss(
            output, batch['decoder_input_ids'], batch['decoder_attention_mask'])
        # loss = loss.item()
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=batch['images'].size(0), sync_dist=True)
        # import ipdb
        # ipdb.set_trace()

        return loss

    def generate(self, images):
        """
        Autoregressively generate a prediction.
        Note that beam_size will be called automatically via args.
        """
        outputs = self(images, mode='sample')
        return outputs

    def _shared_eval_step(self, batch, batch_idx, ):
        # print(batch['images'].size())
        outputs = self.generate(batch['images'])
        generated_report = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return generated_report

    def validation_step(self, batch, batch_idx):
        generated_report = self._shared_eval_step(batch, batch_idx)
        # Log reports
        self.val_report_logger.update(generated_report, dicom_ids=batch['ids'])
        # # Evaluate:
        self.val_chexbert_metrics.update(
            generated_report, batch['report'], ids=batch['ids'])
        self.val_coco_metrics.update(
            generated_report, batch['report'], ids=batch['ids'])

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """
        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()
        scores = {}
        output = self.val_chexbert_metrics.compute()
        scores.update(output)
        self.val_chexbert_metrics.reset()
        output = self.val_coco_metrics.compute()
        scores.update(output)
        self.val_coco_metrics.reset()

        self.log_dict({f'val_{k}': v for k, v in scores.items()}, prog_bar=True,
                      on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int):
        generated_report = self._shared_eval_step(batch, batch_idx)
        # Log reports:
        self.test_report_logger.update(
            generated_report, dicom_ids=batch['ids'])
        # # Evaluate:
        self.val_chexbert_metrics.update(
            generated_report, batch['report'], ids=batch['ids'])
        self.test_coco_metrics.update(
            generated_report, batch['report'], ids=batch['ids'])

    def on_test_epoch_end(self, ):
        # compute
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()
        scores = {}
        output = self.val_chexbert_metrics.compute()
        scores.update(output)
        self.val_chexbert_metrics.reset()
        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        self.log_dict({f'test_{k}': v for k, v in scores.items()}, prog_bar=True,
                      on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        grouped_parameters = [
            {"params": self.encoder.parameters(), 'lr': self.args.lr_ve},
            {"params": self.encoder_decoder.parameters(), 'lr': self.args.lr_ed},
        ]

        optimizer = torch.optim.Adam(grouped_parameters,
                                     weight_decay=self.args.weight_decay,
                                     amsgrad=self.args.amsgrad)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            self.args.step_size,
            self.args.gamma
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_special_all_special_tokens(tokenizer):
        tokenizer.add_special_tokens({'bos_token': '[BOS]',
                                      'eos_token': '[EOS]',
                                      #   'pad_token': '[PAD]'
                                      })
        return tokenizer


# if __name__=='__main__':
#     from merg.tools.path import DATAPATH

#     iu = DATAPATH['iu_xray']
#     annotation_file, dataset_dir = iu['annotation_file'], iu['dataset_dir']

#     r2g = R2Gen(
#         dataset_name='iu_xray',
#         annotation_file=annotation_file,
#         dataset_dir=dataset_dir
#     )
#     import ipdb
#     ipdb.set_trace()
