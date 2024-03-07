from typing import Any, Dict
import os
import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from transformers import AutoTokenizer
from biovlp.datasets.base_dataset import custom_collate_fn
from biovlp.datasets.datamodule import DataModule
from biovlp.datasets.mimic_cxr_dataset import MIMICCXRDataset
from biovlp.datasets.iu_xray_dataset import IUXrayDataset
from biovlp.metrics.report_logger import ReportLogger
from biovlp.metrics.coco import COCOCaptionMetrics
from biovlp.metrics.chexbert import CheXbertMetrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../../")


class BaseLightningModule(LightningModule):
    '''
    Base model for report generation
    '''

    def __init__(self,
                 dataset_name: str,
                 annotation_file: str,
                 dataset_dir: str,
                 exp_log_dir: str,
                 language_model: str,
                 train_data_pct: float = 1.,
                 max_length: int = 128,
                 num_beams: int = 3,
                 batch_size: int = 16,
                 image_size: int = 512,
                 num_workers: int = 8,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 0.05,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.language_model = language_model
        self.annotation_file = annotation_file
        self.dataset_dir = dataset_dir
        self.train_data_pct = train_data_pct
        self.max_length = max_length
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exp_log_dir = exp_log_dir

        self.tokenizer = self.setup_tokenizer()
        self.datamodule = self.setup_datamodule()
        self.setup_report_generation_metrics()

    def setup_report_generation_metrics(self):
        self.val_report_logger = ReportLogger(
            exp_dir=self.exp_log_dir, split='val_reports')
        self.test_report_logger = ReportLogger(
            exp_dir=self.exp_log_dir, split='test_reports')

        self.val_coco_metrics = COCOCaptionMetrics(
            metrics=["bleu", "cider", "rouge"])
        self.test_coco_metrics = COCOCaptionMetrics(
            metrics=["bleu", "cider", "meteor", "rouge"])

        # CheXbert classification metrics:
        self.val_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path=os.path.join(
                REPO_ROOT_DIR, 'pretrained/chexbert/chexbert.pth'),
            mbatch_size=self.batch_size,
            exp_dir=self.exp_log_dir,
        )
        self.test_chexbert_metrics = CheXbertMetrics(
            bert_path='bert-base-uncased',
            checkpoint_path=os.path.join(
                REPO_ROOT_DIR, 'pretrained/chexbert/chexbert.pth'),
            mbatch_size=self.batch_size,
            exp_dir=self.exp_log_dir,
        )

    def setup_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.language_model, trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        return tokenizer

    def setup_datamodule(self):
        '''
        define the datamodule for the model
        '''
        if self.dataset_name == "mimic_cxr":
            dataset = MIMICCXRDataset
        elif self.dataset_name == "iu_xray":
            dataset = IUXrayDataset
        else:
            raise NotImplementedError

        datamodule = DataModule(
            dataset=dataset,
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

        return datamodule

    def forward(self, images, input_ids, attention_mask):
        '''
        Forward step of an encoder-decoder model.
        Output the logits of each token.
        '''
        raise NotImplementedError

    def generate(self, num_beams, images):
        '''
        Generate report given an image.
        '''
        raise NotImplementedError

    def training_step(self, batch: Dict, batch_idx: int):
        y_hat = self(
            batch["images"], batch["decoder_input_ids"], batch["decoder_attention_mask"]
        )

        # Loss:
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]).contiguous(), batch['label_ids'], ignore_index=self.tokenizer.pad_token_id,
        )

        # Logging:
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True,
                      prog_bar=True, batch_size=y_hat.shape[0], sync_dist=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        output_ids = self.generate(self.num_beams, batch["images"])

        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        # Log reports:
        self.val_report_logger.update(generated, dicom_ids=batch['ids'])

        # Evaluate:
        self.val_chexbert_metrics.update(
            generated, batch['report'], ids=batch['ids'])
        self.val_coco_metrics.update(
            generated, batch['report'], ids=batch['ids'])

        if batch_idx == 0:
            print("=" * 10)
            print("generted reports: ")
            print("\n".join(generated[:5]))
            print("="*10)
            print("ground truth reports: ")
            print("\n".join(batch["report"][:5]))
            print("="*10)

    def on_validation_epoch_end(self):

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

    def test_step(self, batch, batch_idx):

        # Beam search:
        output_ids = self.generate(self.num_beams, batch['images'])

        # Generated report:
        generated = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)

        # Log reports:
        self.test_report_logger.update(generated, dicom_ids=batch['ids'])

        # Evaluate:
        self.test_chexbert_metrics.update(
            generated, batch['report'], ids=batch['ids'])
        self.test_coco_metrics.update(
            generated, batch['report'], ids=batch['ids'])

    def on_test_epoch_end(self):

        # Save reports:
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()

        scores = {}

        output = self.test_chexbert_metrics.compute()
        scores.update(output)
        self.test_chexbert_metrics.reset()

        output = self.test_coco_metrics.compute()
        scores.update(output)
        self.test_coco_metrics.reset()

        self.log_dict({f'test_{k}': v for k, v in scores.items()}, prog_bar=True,
                      on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate,
                                 weight_decay=self.weight_decay)
