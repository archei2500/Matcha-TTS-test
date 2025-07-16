import torchaudio
import torch
from speechbrain.inference.speaker import EncoderClassifier


class ECAPAService:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", device=None):
        self.model_source = model_source
        self._device = device
        self._classifier = None

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

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Encode a waveform into an ECAPA speaker embedding.
        waveform shape: (B, T), (B, 1, T), or (T,)
        Returns: embedding tensor of shape (B, D) or (D,)
        """
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)  # (1, 1, T)
            elif waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)  # (B, 1, T)
            # (B, 1, T)
            waveform = waveform.to(self.device)
            emb = self.classifier.encode_batch(waveform).squeeze(1)  # (B, D)
            return emb.cpu()

    def encode_file(self, filepath: str) -> torch.Tensor:
        """
        Load waveform from file and encode to ECAPA embedding.
        Returns: embedding tensor of shape (1, D)
        """
        signal, fs = torchaudio.load(filepath)
        return self.encode(signal.unsqueeze(0))  # (1, T)
