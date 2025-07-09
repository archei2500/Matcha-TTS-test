from matcha.data.text_mel_datamodule import TextMelDataModule, TextMelDataset, parse_filelist, TextMelBatchCollate

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse

import pandas as pd

# сначала всё меняем, потом перепроверяем, чтобы были эти везде изменённые функции
# нужно преобразовать для новой структуры
class TextMelECAPADataset(TextMelDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = pd.read_csv(self.hparams.filelist_path, sep="|", names=["utt_id", "text", "audio_path", "emb_path", "speaker_id"])
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.metadata["speaker_id"].unique())}
        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)  # shuffle equivalent

    def get_datapoint(self, index):
        row = self.metadata.iloc[index]
        base_data = super().get_datapoint([row["audio_path"], str(self.speaker_to_idx[row["speaker_id"]]), row["text"]])
        base_data["ecapa_emb"] = torch.load(row["emb_path"])

        return base_data

    def __getitem__(self, index):
        datapoint = self.get_datapoint(index)
        return datapoint

    def __len__(self):
        return len(self.metadata)


class TextMelECAPABatchCollate(TextMelBatchCollate):
    def __call__(self, batch):
        # Фильтруем только нужные ключи для родителя
        base_batch = [{k: v for k, v in item.items() if k not in ["ecapa_emb"]} for item in batch]
        batch_data = super().__call__(base_batch)

        # Добавляем ECAPA
        if "ecapa_emb" in batch[0]:
            batch_data["ecapa_emb"] = torch.stack([item["ecapa_emb"] for item in batch])

        return batch_data
