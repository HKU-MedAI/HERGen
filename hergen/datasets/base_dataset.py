import json
import os
import random
import ipdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as Dataset_pt
from datasets import Dataset as Dataset_hg
from PIL import Image
from hergen.datasets.transforms import get_transforms


class BaseDataset(Dataset_pt):
    def __init__(self,
                 annotation_file: str,
                 dataset_dir: str,
                 split: str,
                 tokenizer,
                 image_size: int,
                 mean: float,
                 std: float,
                 max_length: int,
                 train_data_pct: float,
                 return_label: bool,
                 return_kg: bool) -> None:
        '''
        For MIMIC-CXR or IU-Xray dataset, the required input are:
        - annotation_file: the metadata "annotation.json". It provides image id -> report pairs.
        - dataset_dir: the root directory of dataset. It contains required images.
        '''
        super().__init__()

        self.annotation_file = annotation_file
        self.max_length = max_length
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.return_label = return_label
        self.return_kg = return_kg
        self.image_transforms = get_transforms(split, image_size=image_size, mean=mean, std=std)

        assert os.path.exists(
            self.annotation_file), f"Annotation file {self.annotation_file} doesn't exist!"
        # Load this medadata
        with open(self.annotation_file) as f:
            examples = json.load(f)[split]

        if split == "train":
            # random sample from training set
            num_samples = int(len(examples) * train_data_pct)
            examples = random.sample(examples, num_samples)

        # create huggingface Dataset class, which is easy to tokenize
        dataset_as_dfs = pd.DataFrame(examples)

        # # FIXME: remove this line later
        # # longitudinal mimic
        # self.min_seq_length = 1
        # subject_cnts = dataset_as_dfs["subject_id"].value_counts()
        # cur_subject_cnts = subject_cnts[subject_cnts >= self.min_seq_length]
        # dataset_as_dfs = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
        #     cur_subject_cnts.index.tolist())]

        def preprocess_chen_tokenizer(report):
            report = self.chen_tokenizer(report)[:self.chen_max_seq_length]
            return self.chen_tokenizer.decode(report[1:])

        dataset_as_dfs["report"] = dataset_as_dfs["report"].apply(
            preprocess_chen_tokenizer)

        dataset = Dataset_hg.from_pandas(dataset_as_dfs)
        self.tokenized_dataset = self.tokenize(dataset)

    def tokenize(self, dataset):
        ''' Tokenize report into input_ids and attention_mask'''
        tokenizer = self.tokenizer

        def tokenize_function(example):
            report = tokenizer.bos_token + \
                example["report"] + tokenizer.eos_token
            return tokenizer(
                report,
                padding="max_length",
                max_length=self.max_length + 1,
                truncation=True,
                return_tensors="pt")

        tokenized_dataset = dataset.map(tokenize_function)

        return tokenized_dataset

    def __len__(self):
        return len(self.tokenized_dataset)

    def load_and_preprocess_image(self, local_image_path: str) -> torch.Tensor:
        ''' Read image and preprocess image'''
        image_path = os.path.join(self.dataset_dir, local_image_path)
        assert os.path.exists(image_path), f"{image_path} doesn't exist!"
        image = self.read_image(image_path)
        if self.image_transforms is not None:
            image = self.image_transforms(image)
            # transformed = self.image_transforms(image=image)
            # image = transformed["image"]  # (3, 512, 512)

        return image

    def prepare_decoder_input(self, example):
        '''
        Create decoder input ids, attention mask and label ids.
        '''
        decoder_input_ids = torch.tensor(example["input_ids"][0])
        decoder_attention_mask = torch.tensor(example["attention_mask"][0])[1:]
        label_ids = decoder_input_ids[1:].detach().clone()
        decoder_input_ids = decoder_input_ids[:-1]
        decoder_input_ids[decoder_input_ids ==
                          self.tokenizer.sep_token_id] = self.tokenizer.pad_token_id

        return decoder_input_ids, decoder_attention_mask, label_ids

    @staticmethod
    def read_image(image_path: str) -> np.ndarray:
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            image = Image.open(image_path).convert("RGB")
            return image
        elif image_path.endswith(".dicom") or image_path.endswith(".dcm"):
            # implemented later
            raise NotImplementedError


def custom_collate_fn(batch):
    ids, image, report, decoder_input_ids, decoder_attention_mask, label_ids = [
    ], [], [], [], [], []
    for example_dict in batch:
        ids.append(example_dict["id"])
        image.append(example_dict["image"])
        report.append(example_dict["report"])
        decoder_input_ids.append(example_dict["decoder_input_ids"])
        decoder_attention_mask.append(example_dict["decoder_attention_mask"])
        label_ids.append(example_dict["label_ids"])
    image = torch.stack(image)
    decoder_input_ids = torch.stack(decoder_input_ids)
    decoder_attention_mask = torch.stack(decoder_attention_mask)
    label_ids = torch.stack(label_ids)

    return {
        "ids": ids,
        "images": image,
        "report": report,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "label_ids": label_ids
    }
