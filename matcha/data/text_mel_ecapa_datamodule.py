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


class TextMelECAPADataModule(TextMelDataModule):
    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        self.trainset = TextMelECAPADataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self.validset = TextMelECAPADataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelECAPABatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelECAPABatchCollate(self.hparams.n_spks),
        )


class TextMelECAPADataset(TextMelDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = pd.read_csv(self.hparams.filelist_path, sep="|", names=["utt_id", "text", "audio_path", "emb_path", "speaker_id"])
        # self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.metadata["speaker_id"].unique())}
        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)  # shuffle equivalent

    def get_datapoint(self, index):
        row = self.metadata.iloc[index]
        # base_data = super().get_datapoint([row["audio_path"], str(self.speaker_to_idx[row["speaker_id"]]), row["text"]])
        base_data = super().get_datapoint([row["audio_path"], int(row["speaker_id"]), row["text"]])
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
