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


class TextMelECAPADataModule(TextMelDataModule):
    def __init__(
        self,
        ecapa_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelECAPADataset(  # pylint: disable=attribute-defined-outside-init
            ecapa_dir=self.ecapa_dir,
            filelist_path=self.hparams.train_filelist_path,
            n_spks=self.hparams.n_spks,
            cleaners=self.hparams.cleaners,
            add_blank=self.hparams.add_blank,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_feats,
            sample_rate=self.hparams.sample_rate,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            f_min=self.hparams.f_min,
            f_max=self.hparams.f_max,
            data_parameters=self.hparams.data_statistics,
            seed=self.hparams.seed,
            load_durations=self.hparams.load_durations,
        )
        self.validset = TextMelECAPADataset(  # pylint: disable=attribute-defined-outside-init
            ecapa_dir=self.ecapa_dir,
            filelist_path=self.hparams.valid_filelist_path,
            n_spks=self.hparams.n_spks,
            cleaners=self.hparams.cleaners,
            add_blank=self.hparams.add_blank,
            n_fft=self.hparams.n_fft,
            n_mels=self.hparams.n_feats,
            sample_rate=self.hparams.sample_rate,
            hop_length=self.hparams.hop_length,
            win_length=self.hparams.win_length,
            f_min=self.hparams.f_min,
            f_max=self.hparams.f_max,
            data_parameters=self.hparams.data_statistics,
            seed=self.hparams.seed,
            load_durations=self.hparams.load_durations,
        )


class TextMelECAPADataset(TextMelDataset):
    def __init__(
        self,
        ecapa_dir,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ecapa_dir = Path(ecapa_dir)

    def get_datapoint(self, filepath_and_text):
        data = super().get_datapoint(filepath_and_text)
        # CHECK
        utt_id = Path(data["filepath"]).stem
        ecapa_path = self.ecapa_dir / f"{utt_id}.pt"
        data["ecapa"] = torch.load(ecapa_path)
        return data


class TextMelECAPABatchCollate(TextMelBatchCollate):
    def __call__(self, batch):
        # Фильтруем только нужные ключи для родителя
        base_batch = [{k: v for k, v in item.items() if k not in ["ecapa"]} for item in batch]
        batch_data = super().__call__(base_batch)

        # Добавляем ECAPA
        if "ecapa" in batch[0]:
            batch_data["ecapa"] = torch.stack([item["ecapa"] for item in batch])

        return batch_data
