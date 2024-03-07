import torch
from typing import Dict
from biovlp.datasets.base_dataset import BaseDataset
from biovlp.datasets.mimic_cxr_chen_tokenizer import TokenizerChen
import ipdb


class MIMICLMDataset(BaseDataset):
    '''
    For the LM dataset, we don't load images.
    '''

    def __init__(self,
                 annotation_file: str,
                 dataset_dir: str,
                 split: str,
                 tokenizer,
                 image_size: int = 512,
                 max_length: int = 128,
                 train_data_pct: float = 1.,
                 return_label: bool = False,
                 return_kg: bool = False) -> None:

        self.dataset_name = "mimic_cxr"
        self.chen_tokenizer = TokenizerChen(
            ann_path=annotation_file,
            threshold=10,
        )
        self.chen_max_seq_length = 100

        super().__init__(annotation_file, dataset_dir, split, tokenizer, image_size,
                         max_length, train_data_pct, return_label, return_kg)

    def __getitem__(self, index) -> Dict:
        '''
        - report: str
        - input_ids: [128]
        - attention_mask: [128]
        '''

        example = self.tokenized_dataset[index]
        decoder_input_ids, decoder_attention_mask, label_ids = self.prepare_decoder_input(
            example)

        example_dict = {
            "id": example["id"],
            "report": example["report"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "label_ids": label_ids
        }

        return example_dict


def custom_lm_collate_fn(batch):
    ids, report, decoder_input_ids, decoder_attention_mask, label_ids = [
    ], [], [], [], []
    for example_dict in batch:
        ids.append(example_dict["id"])
        report.append(example_dict["report"])
        decoder_input_ids.append(example_dict["decoder_input_ids"])
        decoder_attention_mask.append(example_dict["decoder_attention_mask"])
        label_ids.append(example_dict["label_ids"])

    decoder_input_ids = torch.stack(decoder_input_ids)
    decoder_attention_mask = torch.stack(decoder_attention_mask)
    label_ids = torch.stack(label_ids)

    return {
        "ids": ids,
        "report": report,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "label_ids": label_ids
    }


class MIMICCXRDataset(BaseDataset):
    def __init__(self,
                 annotation_file: str,
                 dataset_dir: str,
                 split: str,
                 tokenizer,
                 image_size: int = 512,
                 max_length: int = 128,
                 train_data_pct: float = 1.,
                 return_label: bool = False,
                 return_kg: bool = False) -> None:

        self.dataset_name = "mimic_cxr"
        self.chen_tokenizer = TokenizerChen(
            ann_path=annotation_file,
            threshold=10,
        )
        self.chen_max_seq_length = 100

        super().__init__(annotation_file, dataset_dir, split, tokenizer, image_size,
                         max_length, train_data_pct, return_label, return_kg)

    def __getitem__(self, index) -> Dict:
        '''
        - image: [3, 512, 512]
        - report: str
        - input_ids: [128]
        - attention_mask: [128]
        '''

        example = self.tokenized_dataset[index]
        image = self.load_and_preprocess_image(example["image_path"][0])

        decoder_input_ids, decoder_attention_mask, label_ids = self.prepare_decoder_input(
            example)

        example_dict = {
            "id": example["id"],
            "image": image,
            "report": example["report"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "label_ids": label_ids
        }

        return example_dict


if __name__ == "__main__":
    from transformers import GPT2TokenizerFast
    language_model = "distilgpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(language_model)
    tokenizer.add_special_tokens(
        {"bos_token": "[BOS]", 'pad_token': '[PAD]'})
    dataset = MIMICCXRDataset(
        annotation_file="/home/r15user2/Documents/CXR_dataset/temporal_CXR/mimic_annotation.json",
        dataset_dir="/home/r15user2/Documents/CXR_dataset/mimic_data/2.0.0/files",
        split="test",
        tokenizer=tokenizer
    )
    print(dataset[0])
    df = dataset.tokenized_dataset.to_pandas()
    df.loc[df["id"] == "ea9b867c-c8a2b175-f813e34d-9ae7229d-23ab7c24", "report"].values
    ipdb.set_trace()
