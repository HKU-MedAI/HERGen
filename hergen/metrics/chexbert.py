import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer
from sklearn.metrics import precision_recall_fscore_support
from hergen.metrics.natural_language import NaturalLanguage

# from hergen.utils import enumerated_save_path

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

EVALUATE_5_CONDITIONS = [
    "cardiomegaly",
    "edema",
    "consolidation",
    "atelectasis",
    "pleural_effusion"
]


CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


class CheXbert(nn.Module):
    def __init__(self, bert_path, checkpoint_path, device, p=0.1):
        super(CheXbert, self).__init__()

        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        config = BertConfig().from_pretrained(bert_path)

        with torch.no_grad():

            self.bert = BertModel(config)
            self.dropout = nn.Dropout(p)

            hidden_size = self.bert.pooler.dense.in_features

            # Classes: present, absent, unknown, blank for 12 conditions + support devices
            self.linear_heads = nn.ModuleList(
                [nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])

            # Classes: yes, no for the 'no finding' observation
            self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

            # Load CheXbert checkpoint
            state_dict = torch.load(checkpoint_path, map_location=device)[
                'model_state_dict']

            new_state_dict = OrderedDict()
            new_state_dict["bert.embeddings.position_ids"] = torch.arange(
                config.max_position_embeddings).expand((1, -1))
            for key, value in state_dict.items():
                if 'bert' in key:
                    new_key = key.replace('module.bert.', 'bert.')
                elif 'linear_heads' in key:
                    new_key = key.replace(
                        'module.linear_heads.', 'linear_heads.')
                new_state_dict[new_key] = value

            self.load_state_dict(new_state_dict, strict=False)

        self.eval()

    def forward(self, reports):

        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace("\n", " ")
            reports[i] = reports[i].replace("\s+", " ")
            reports[i] = reports[i].replace("\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()

        with torch.no_grad():

            tokenized = self.tokenizer(
                reports, padding='longest', return_tensors="pt")
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

            last_hidden_state = self.bert(**tokenized)[0]

            cls = last_hidden_state[:, 0, :]
            cls = self.dropout(cls)

            predictions = []
            for i in range(14):
                predictions.append(self.linear_heads[i](cls).argmax(dim=1))

        return torch.stack(predictions, dim=1)


def enumerated_save_path(save_dir, save_name, extension):
    save_path = os.path.join(save_dir, save_name + extension)
    assert '.' in extension, 'No period in extension.'
    if os.path.isfile(save_path):
        count = 2
        while True:
            save_path = os.path.join(
                save_dir, save_name + "_" + str(count) + extension)
            count += 1
            if not os.path.isfile(save_path):
                break

    return save_path


class CheXbertMetrics(NaturalLanguage):

    is_differentiable = False
    full_state_update = False

    def __init__(
        self,
        bert_path,
        checkpoint_path,
        mbatch_size=16,
        save_class_scores=False,
        save_outputs=False,
        exp_dir=None,
    ):
        super().__init__(dist_sync_on_step=False)

        self.bert_path = bert_path
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.save_class_scores = save_class_scores
        self.save_outputs = save_outputs
        self.exp_dir = exp_dir

    def mini_batch(self, iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def compute(self):

        chexbert = CheXbert(
            bert_path=self.bert_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        ).to(self.device)

        table = {'chexbert_y_hat': [], 'chexbert_y': [],
                 'y_hat': [], 'y': [], 'ids': []}
        for i in self.mini_batch(self.pairs, self.mbatch_size):
            y_hat, y, ids = zip(*i)
            table['chexbert_y_hat'].extend(
                [i + [j] for i, j in zip(chexbert(list(y_hat)).tolist(), list(ids))])
            table['chexbert_y'].extend(
                [i + [j] for i, j in zip(chexbert(list(y)).tolist(), list(ids))])

            # FIXME: add micro-5 later
            table['y_hat'].extend(y_hat)
            table['y'].extend(y)
            table['ids'].extend(ids)

        if torch.distributed.is_initialized():  # If DDP

            chexbert_y_hat_gathered = [None] * \
                torch.distributed.get_world_size()
            chexbert_y_gathered = [None] * torch.distributed.get_world_size()
            y_hat_gathered = [None] * torch.distributed.get_world_size()
            y_gathered = [None] * torch.distributed.get_world_size()
            ids_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(
                chexbert_y_hat_gathered, table['chexbert_y_hat'])
            torch.distributed.all_gather_object(
                chexbert_y_gathered, table['chexbert_y'])
            torch.distributed.all_gather_object(y_hat_gathered, table['y_hat'])
            torch.distributed.all_gather_object(y_gathered, table['y'])
            torch.distributed.all_gather_object(ids_gathered, table['ids'])

            table['chexbert_y_hat'] = [
                j for i in chexbert_y_hat_gathered for j in i]
            table['chexbert_y'] = [j for i in chexbert_y_gathered for j in i]
            table['y_hat'] = [j for i in y_hat_gathered for j in i]
            table['y'] = [j for i in y_gathered for j in i]
            table['ids'] = [j for i in ids_gathered for j in i]

        columns = CONDITIONS + ['ids']
        df_y_hat = pd.DataFrame.from_records(
            table['chexbert_y_hat'], columns=columns)
        df_y = pd.DataFrame.from_records(table['chexbert_y'], columns=columns)

        df_y_hat = df_y_hat.drop_duplicates(subset=['ids'])
        df_y = df_y.drop_duplicates(subset=['ids'])

        df_y_hat = df_y_hat.drop(['ids'], axis=1)
        df_y = df_y.drop(['ids'], axis=1)

        df_y_hat_5 = df_y_hat.loc[:, EVALUATE_5_CONDITIONS]
        df_y_5 = df_y.loc[:, EVALUATE_5_CONDITIONS]

        def convert_label(label: int):
            if label == 2:
                return 0
            elif label == 3:
                return 1
            else:
                return label

        # convert label for each condition
        df_y_hat_5 = df_y_hat_5.applymap(convert_label)
        df_y_5 = df_y_5.applymap(convert_label)

        precision, recall, f1, _ = precision_recall_fscore_support(
            df_y_hat_5.T.values.reshape(-1), df_y_5.T.values.reshape(-1), average='binary')

        precision_5, recall_5, f1_5, _ = precision_recall_fscore_support(df_y_hat_5.values, df_y_5.values)
        precision_5, recall_5, f1_5 = [], [], []
        for i in range(5):
            p, r, f, _ = precision_recall_fscore_support(df_y_hat_5.iloc[:, i].values,
                                                         df_y_5.iloc[:,
                                                                     i].values,
                                                         average='binary')
            precision_5.append(p)
            recall_5.append(r)
            f1_5.append(f)

        df_y_hat = (df_y_hat == 1)
        df_y = (df_y == 1)
        tp = (df_y_hat * df_y).astype(float)
        fp = (df_y_hat * ~df_y).astype(float)
        fn = (~df_y_hat * df_y).astype(float)

        tp_cls = tp.sum()
        fp_cls = fp.sum()
        fn_cls = fn.sum()

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_class = (tp_cls / (tp_cls + fp_cls)).fillna(0)
        recall_class = (tp_cls / (tp_cls + fn_cls)).fillna(0)
        f1_class = (tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls))).fillna(0)

        scores = {
            'ce_precision_macro': precision_class.mean(),
            'ce_recall_macro': recall_class.mean(),
            'ce_f1_macro': f1_class.mean(),
            'ce_precision_micro': tp_cls.sum() / (tp_cls.sum() + fp_cls.sum()),
            'ce_recall_micro': tp_cls.sum() / (tp_cls.sum() + fn_cls.sum()),
            'ce_f1_micro': tp_cls.sum() / (tp_cls.sum() + 0.5 * (fp_cls.sum() + fn_cls.sum())),
            'ce_precision_example': (tp_eg / (tp_eg + fp_eg)).fillna(0).mean(),
            'ce_recall_example': (tp_eg / (tp_eg + fn_eg)).fillna(0).mean(),
            'ce_f1_example': (tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).fillna(0).mean(),
            'ce_num_examples': float(len(df_y_hat)),
            'ce_precision_micro_5': precision,
            'ce_recall_micro_5': recall,
            'ce_f1_micro_5': f1,
            'ce_precision_macro_5': np.mean(precision_5),
            'ce_recall_macro_5': np.mean(recall_5),
            'ce_f1_macro_5': np.mean(f1_5),
        }

        if self.save_class_scores:
            save_path = enumerated_save_path(
                self.exp_dir, 'ce_class_metrics', '.csv')
            class_scores_dict = {
                **{'ce_precision_' + k: v for k, v in precision_class.to_dict().items()},
                **{'ce_recall_' + k: v for k, v in recall_class.to_dict().items()},
                **{'ce_f1_' + k: v for k, v in f1_class.to_dict().items()},
            }
            pd.DataFrame(class_scores_dict, index=['i', ]).to_csv(
                save_path, index=False)

        if self.save_outputs:

            def save():
                df = pd.DataFrame(table)
                df.chexbert_y_hat = [i[:-1] for i in df.chexbert_y_hat]
                df.chexbert_y = [i[:-1] for i in df.chexbert_y]
                df.to_csv(
                    os.path.join(self.exp_dir, 'chexbert_outputs_' +
                                 time.strftime("%d-%m-%Y_%H-%M-%S") + '.csv'),
                    index=False,
                    sep=';',
                )
            if not torch.distributed.is_initialized():
                save()
            elif torch.distributed.get_rank() == 0:
                save()

        return scores
