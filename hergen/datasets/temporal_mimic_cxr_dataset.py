from typing import Dict
import pandas as pd
import torch
import ipdb
from biovlp.datasets.mimic_cxr_dataset import MIMICCXRDataset


pd.set_option('mode.chained_assignment', None)


class TemporalMIMICCXRDataset(MIMICCXRDataset):
    ''' Temporal MIMIC-CXR dataset classs '''

    def __init__(self,
                 annotation_file: str,
                 dataset_dir: str,
                 split: str,
                 tokenizer,
                 image_size: int = 512,
                 max_length: int = 128,
                 min_seq_length: int = 1,
                 max_seq_length: int = 5,
                 train_data_pct: float = 1,
                 return_label: bool = False,
                 return_kg: bool = False) -> None:

        super().__init__(annotation_file, dataset_dir, split, tokenizer, image_size,
                         max_length, train_data_pct, return_label, return_kg)

        dataset_as_dfs = pd.DataFrame(self.tokenized_dataset)
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length

        subject_cnts = dataset_as_dfs["subject_id"].value_counts()
        cur_subject_cnts = subject_cnts[subject_cnts >= self.min_seq_length]
        dataset_as_dfs = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
            cur_subject_cnts.index.tolist())]

        subject_cnts = dataset_as_dfs["subject_id"].value_counts()

        subject_cnts = subject_cnts[subject_cnts > self.max_seq_length]
        dataset_part_1 = dataset_as_dfs.loc[~dataset_as_dfs["subject_id"].isin(
            subject_cnts.index.tolist())]
        dataset_part_2 = dataset_as_dfs.loc[dataset_as_dfs["subject_id"].isin(
            subject_cnts.index.tolist())]

        # we split all patient sequence which have more than 5 images into several sequences
        if len(dataset_part_2) > 0:
            new_dfs = []
            for subject_id in dataset_part_2["subject_id"].unique():
                sub_df = dataset_part_2.loc[dataset_part_2["subject_id"]
                                            == subject_id]
                sub_df.sort_values(
                    by=["subject_id", "StudyDate", "StudyTime"], inplace=True)

                num_images = len(sub_df)
                for i in range(num_images // self.max_seq_length + 1):
                    cur_sub_df = sub_df.iloc[i *
                                             self.max_seq_length: (i + 1) * self.max_seq_length]
                    cur_sub_df["subject_id"] = str(subject_id) + "_" + str(i)
                    new_dfs.append(cur_sub_df)

            dataset_part_2 = pd.concat(new_dfs, axis=0)
            self.dataset_as_dfs = pd.concat(
                [dataset_part_1, dataset_part_2], axis=0)
        else:
            self.dataset_as_dfs = dataset_part_1
        self.dataset_as_dfs.reset_index(inplace=True, drop=True)

        self.unique_ptids = self.dataset_as_dfs["subject_id"].unique().tolist()
        self.dataset_as_dfs["subject_id"].value_counts()

    def __len__(self):
        return len(self.unique_ptids)

    def __getitem__(self, index) -> Dict:
        '''
        Each samles is a dict:
        - ids
        - images 
        - report
        - decoder_input_ids
        - decoder_attention_mask
        - label_ids
        - batch_mask
        - study_date
        '''
        ptid = self.unique_ptids[index]
        sub_df = self.dataset_as_dfs.loc[self.dataset_as_dfs["subject_id"] == ptid]
        sub_df.sort_values(by=["StudyDate", "StudyTime"], inplace=True)
        sub_df.reset_index(inplace=True, drop=True)

        id_list, image_list, decoder_input_ids_list, decoder_attention_mask_list, label_ids_list, report_list, study_date_list = [
        ], [], [], [], [], [], []
        for example in sub_df.to_dict(orient="records"):
            image = self.load_and_preprocess_image(example["image_path"][0])
            image_list.append(image)
            decoder_input_ids, decoder_attention_mask, label_ids = self.prepare_decoder_input(
                example)
            decoder_input_ids_list.append(decoder_input_ids)
            decoder_attention_mask_list.append(decoder_attention_mask)
            label_ids_list.append(label_ids)
            report_list.append(example["report"])
            id_list.append(example["id"])
            study_date_list.append(example["StudyDate"])

        image_list = torch.stack(image_list)
        decoder_input_ids_list = torch.stack(decoder_input_ids_list)
        decoder_attention_mask_list = torch.stack(decoder_attention_mask_list)
        label_ids_list = torch.stack(label_ids_list)

        # now we need to pad samples to make sure they can appear in the same batch
        batch_mask = torch.tensor([False] * self.max_seq_length)
        batch_mask[:len(id_list)] = True

        if len(id_list) < self.max_seq_length:
            # pad several nan samples
            pad_length = self.max_seq_length - len(id_list)
            # id_list.extend([""] * pad_length)
            # report_list.extend([""] * pad_length)

            pad_value = study_date_list[0]
            study_date_list.extend([pad_value] * pad_length)

            # pad images
            _, c, h, w = image_list.shape
            pad_image = torch.zeros(pad_length, c, h, w)
            image_list = torch.cat([image_list, pad_image], dim=0)

            # pad decoder_input_ids
            d = decoder_input_ids_list.shape[-1]
            pad_decoder_input_ids = torch.zeros(pad_length, d).long()
            decoder_input_ids_list = torch.cat(
                [decoder_input_ids_list, pad_decoder_input_ids], dim=0)
            decoder_attention_mask_list = torch.cat(
                [decoder_attention_mask_list, pad_decoder_input_ids], dim=0)
            label_ids_list = torch.cat(
                [label_ids_list, pad_decoder_input_ids], dim=0)

        study_date_list = torch.tensor(study_date_list).float()
        pad_value = study_date_list[0]
        study_date_list = study_date_list - pad_value
        study_date_list = study_date_list / 1000.
        example_dict = {
            "id": id_list,
            "image": image_list,
            "report": report_list,
            "decoder_input_ids": decoder_input_ids_list,
            "decoder_attention_mask": decoder_attention_mask_list,
            "label_ids": label_ids_list,
            "batch_mask": batch_mask,
            "study_date": study_date_list
        }

        return example_dict


def temporal_collate_fn(batch):
    ids, image, report, decoder_input_ids, decoder_attention_mask, label_ids, batch_mask, study_date = [
    ], [], [], [], [], [], [], []
    for example_dict in batch:
        ids.append(example_dict["id"])
        image.append(example_dict["image"])
        report.append(example_dict["report"])
        decoder_input_ids.append(example_dict["decoder_input_ids"])
        decoder_attention_mask.append(example_dict["decoder_attention_mask"])
        label_ids.append(example_dict["label_ids"])
        batch_mask.append(example_dict["batch_mask"])
        study_date.append(example_dict["study_date"])

    image = torch.stack(image)
    decoder_input_ids = torch.stack(decoder_input_ids)
    decoder_attention_mask = torch.stack(decoder_attention_mask)
    label_ids = torch.stack(label_ids)
    batch_mask = torch.stack(batch_mask)
    study_date = torch.stack(study_date)

    return {
        "ids": ids,
        "images": image,
        "report": report,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "label_ids": label_ids,
        "batch_mask": batch_mask,
        "study_date": study_date
    }


if __name__ == "__main__":
    from transformers import GPT2TokenizerFast
    language_model = "distilgpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(language_model)
    tokenizer.add_special_tokens(
        {"bos_token": "[BOS]", 'pad_token': '[PAD]'})
    dataset = TemporalMIMICCXRDataset(
        annotation_file="/home/r15user2/Documents/CXR_dataset/temporal_CXR/mimic_annotation.json",
        dataset_dir="/home/r15user2/Documents/CXR_dataset/mimic_data/2.0.0/files",
        split="train",
        train_data_pct=1,
        tokenizer=tokenizer
    )
    data = dataset[0]
    ipdb.set_trace()
