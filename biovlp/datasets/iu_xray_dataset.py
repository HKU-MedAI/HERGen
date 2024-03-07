import re
from typing import Dict
import torch
from biovlp.datasets.base_dataset import BaseDataset
from biovlp.datasets.iu_xray_chen_tokenizer import TokenizerChen


class IUXrayDataset(BaseDataset):
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

        self.dataset_name = "iu_xray"
        self.chen_tokenizer = TokenizerChen(
            ann_path=annotation_file,
            threshold=3,
        )
        self.chen_max_seq_length = 60

        super().__init__(annotation_file, dataset_dir, split, tokenizer, image_size,
                         max_length, train_data_pct, return_label, return_kg)

    def __getitem__(self, index) -> Dict:
        '''
        - id: str
        - image: [3, 512, 512]
        - report: str
        - decoder_input_ids: [128]
        - decoder_attention_mask: [128]
        - label_ids: [128]
        '''

        example = self.tokenized_dataset[index]
        # Load frontal view image
        image_1 = self.load_and_preprocess_image(example["image_path"][0])
        image_2 = self.load_and_preprocess_image(example["image_path"][1])
        image = torch.stack((image_1, image_2), 0)

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
    dataset = IUXrayDataset(
        annotation_file="/home/r15user2/Documents/CXR_dataset/knowledge_graph/iu_annotation.json",
        dataset_dir="/home/r15user2/Documents/CXR_dataset/IU_Xray/images_dcl",
        split="train",
        tokenizer=tokenizer
    )
    example_dict = dataset[0]
    print(example_dict["image"].shape)
