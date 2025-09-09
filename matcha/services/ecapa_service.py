import torchaudio
import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.dataio.preprocess import AudioNormalizer


class ECAPAService:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", device=None):
        self.model_source = model_source
        self._device = device
        self._classifier = None
        self.audio_norm = AudioNormalizer()
        self.target_sample_rate = 16000

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        if self._classifier is not None:
            self._classifier.to(value)

    @property
    def classifier(self):
        if self._classifier is None:
            self._initialize_classifier()
        return self._classifier

    def _initialize_classifier(self):
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[ECAPAService] Auto-selected device: {self._device}")
        print(f"[ECAPAService] Loading ECAPA model from {self.model_source}...")
        self._classifier = EncoderClassifier.from_hparams(
            source=self.model_source,
            run_opts={"device": self._device},
        )
        print("[ECAPAService] ECAPA model loaded.")

    def _preprocess_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        # Если waveform имеет размерность (T,) -> (1, 1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        # Если (B, T) -> (B, 1, T)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Нормализация громкости
        waveform = self.audio_norm(waveform, self.target_sample_rate)

        # Обрезка/паддинг до ~3 секунд
        target_length = 48000
        if waveform.size(-1) > target_length:
            waveform = waveform[..., :target_length]
        else:
            pad = target_length - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        return waveform.to(self.device)

    # def encode(self, waveform: torch.Tensor) -> torch.Tensor:
    #     """
    #     Encode a waveform into an ECAPA speaker embedding.
    #     waveform shape: (B, T), (B, 1, T), or (T,)
    #     Returns: embedding tensor of shape (B, D) or (D,)
    #     """
    #     with torch.no_grad():
    #         if waveform.dim() == 1:
    #             waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    #         elif waveform.dim() == 2:
    #             waveform = waveform.unsqueeze(1)  # (B, 1, T)
    #         # Нормализация громкости
    #         waveform = self.audio_norm(waveform, self.target_sample_rate)
    #         waveform = waveform.to(self.device)
    #         emb = self.classifier.encode_batch(waveform).squeeze(1)  # (B, D)
    #         return emb.cpu()

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to ECAPA embedding.
        Args:
            waveform: (B, T) or (T,)
        Returns:
            embedding: (B, D) or (D,)
        """
        waveform = self._preprocess_waveform(waveform)

        with torch.inference_mode():
            embeddings = []
            for wav in waveform.unbind(0):
                emb = self.classifier.encode_batch(wav).squeeze()
                embeddings.append(emb)
            return torch.stack(embeddings) if len(embeddings) > 1 else embeddings[0]

        # OLD
        # with torch.no_grad():
        #     # Получаем эмбеддинги для каждого элемента в батче
        #     embeddings = []
        #     for wav in waveform.unbind(0):  # Итерируем по батчу
        #         emb = self.classifier.encode_batch(wav).squeeze()  # (D,)
        #         embeddings.append(emb)
        #
        #     if len(embeddings) > 1:
        #         return torch.stack(embeddings)  # (B, D)
        #     return embeddings[0]  # (D,)

    # def encode_file(self, filepath: str) -> torch.Tensor:
    #     """
    #     Load waveform from file and encode to ECAPA embedding.
    #     Returns: embedding tensor of shape (1, D)
    #     """
    #     signal, fs = torchaudio.load(filepath)
    #     return self.encode(signal.unsqueeze(0))  # (1, T)

    def encode_file(self, filepath: str) -> torch.Tensor:
        """Load and encode audio file"""
        signal, fs = torchaudio.load(filepath)

        # 09.09
        # Конвертируем стерео в моно
        if signal.dim() == 2 and signal.size(0) == 2:
            signal = signal.mean(dim=0, keepdim=True)  # [1, T]

        if fs != self.target_sample_rate:
            signal = torchaudio.functional.resample(signal, fs, self.target_sample_rate)
        # return self.encode(signal)
        emb = self.encode(signal)  # теперь всегда возвращает [B, 192]

        # Для одного файла гарантируем [1, 192]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)

        return emb

        # # Конвертируем стерео в моно
        # if signal.dim() == 2 and signal.size(0) == 2:
        #     signal = signal.mean(dim=0, keepdim=True)  # [1, T]
        #
        # if fs != self.target_sample_rate:
        #     signal = torchaudio.functional.resample(signal, fs, self.target_sample_rate)
