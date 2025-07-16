import torch
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN


class VocoderService:
    def __init__(self, vocoder_name, checkpoint_path, device=None):
        self.vocoder_name = vocoder_name
        self.checkpoint_path = checkpoint_path
        self._device = device
        self._vocoder = None
        self._denoiser = None
        self._denoiser_strength = 0.00025

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        if self._vocoder is not None:
            self._vocoder.to(value)

    @property
    def vocoder(self):
        if self._vocoder is None:
            self._initialize_vocoder()
        return self._vocoder

    @property
    def denoiser(self):
        if self._denoiser is None:
            self._initialize_denoiser()
        return self._denoiser

    def _initialize_vocoder(self):
        """Load vocoder model based on configuration."""
        print(f"[VocoderService] Loading vocoder '{self.vocoder_name}'...")

        if self._device is None:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[VocoderService] Auto-selected device: {self._device}")

        if self.vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
            self._vocoder = self._load_hifigan()
        else:
            raise NotImplementedError(f"Vocoder '{self.vocoder_name}' not supported.")

        print(f"[VocoderService] Vocoder loaded on {self._device}")

    # def _load_vocoder(self):
    #     print(f"[VocoderService] Loading vocoder '{self.vocoder_name}'...")
    #     if self.vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
    #         self.vocoder = self._load_hifigan()
    #     else:
    #         raise NotImplementedError(f"Vocoder '{self.vocoder_name}' not supported.")
    #
    #     self.denoiser = Denoiser(self.vocoder, mode="zeros")
    #     print(f"[VocoderService] Vocoder '{self.vocoder_name}' loaded.")

    def _load_hifigan(self):
        h = AttrDict(v1)
        model = HiFiGAN(h).to(self._device)
        state_dict = torch.load(self.checkpoint_path, map_location=self._device)["generator"]
        model.load_state_dict(state_dict)
        model.eval()
        model.remove_weight_norm()
        return model

    def _initialize_denoiser(self):
        """Initialize denoiser wrapper for the vocoder."""
        if self._vocoder is None:
            self._initialize_vocoder()
        print("[VocoderService] Initializing denoiser...")
        self._denoiser = Denoiser(self._vocoder, mode="zeros")

    def vocoder_infer(self, mel, denoiser_strength=None):
        """
        Convert mel spectrogram to waveform using vocoder and denoiser.
        """
        if denoiser_strength is None:
            denoiser_strength = self._denoiser_strength

        with torch.no_grad():
            mel = mel.to(self.device)
            audio = self.vocoder(mel).clamp(-1, 1)
            if denoiser_strength > 0 and self.denoiser is not None:
                audio = self.denoiser(audio.squeeze(), strength=denoiser_strength)
            return audio.cpu().squeeze()

    def to(self, device: str):
        """Move all components to specified device."""
        self.device = device
        return self

