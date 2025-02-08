import os
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DataModule(LightningDataModule):
    '''
    DataModule for report generation and pretraining dataset
    '''

    def __init__(self, dataset, tokenizer, annotation_file, dataset_dir, collate_fn, max_length=128,
                 train_data_pct=1., batch_size=8, num_workers=4, image_size=512):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.annotation_file = annotation_file
        self.dataset_dir = dataset_dir
        self.max_length = max_length
        self.train_data_pct = train_data_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.collate_fn = collate_fn

    def train_dataloader(self):

        dataset = self.dataset(
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            split="train",
            tokenizer=self.tokenizer,
            train_data_pct=self.train_data_pct,
            max_length=self.max_length,
            image_size=self.image_size
        )

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):

        dataset = self.dataset(
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            split="val",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            image_size=self.image_size
        )

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):

        dataset = self.dataset(
            annotation_file=self.annotation_file,
            dataset_dir=self.dataset_dir,
            split="test",
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            image_size=self.image_size
        )

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )


if __name__ == "__main__":
    from hergen.datasets.mimic_cxr_dataset import MIMICCXRDataset
    from hergen.datasets.temporal_mimic_cxr_dataset import TemporalMIMICCXRDataset, temporal_collate_fn
    from transformers import GPT2TokenizerFast
    language_model = "distilgpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(language_model)
    tokenizer.add_special_tokens(
        {"bos_token": "[BOS]", 'pad_token': '[PAD]'})
    annotation_file = "/home/r15user2/Documents/CXR_dataset/temporal_CXR/mimic_annotation.json"
    dataset_dir = "/home/r15user2/Documents/CXR_dataset/mimic_data/2.0.0/files"
    dm = DataModule(TemporalMIMICCXRDataset, tokenizer, annotation_file, dataset_dir,
                    batch_size=4, collate_fn=temporal_collate_fn)
    for batch in dm.test_dataloader():
        break

    import ipdb
    ipdb.set_trace()
