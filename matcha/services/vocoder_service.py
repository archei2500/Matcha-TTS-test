import torch
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir
from contextlib import nullcontext


class VocoderService:
    VOCODER_URLS = {
        "hifigan_T2_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1",
        "hifigan_univ_v1": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/g_02500000",
    }

    def __init__(self, vocoder_name, checkpoint_path=None, device=None):
        self.vocoder_name = vocoder_name
        self.checkpoint_path = checkpoint_path
        self._device = device
        self._vocoder = None
        self._denoiser = None
        self._denoiser_strength = 0.00025

        if checkpoint_path is None:
            save_dir = get_user_data_dir()
            self.checkpoint_path = save_dir / f"vocoder_{vocoder_name}"
            assert_model_downloaded(self.checkpoint_path, self.VOCODER_URLS[vocoder_name])
        else:
            self.checkpoint_path = Path(checkpoint_path)

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

        # new 25.08
        use_denoiser = (denoiser_strength > 0.0)

        with torch.inference_mode():
            mel = mel.to(self.device)
            audio = self.vocoder(mel).clamp(-1, 1)

            if use_denoiser and self.denoiser is not None:
                audio = self.denoiser(audio.squeeze(1), strength=denoiser_strength)

            return audio.detach().cpu().squeeze()

        # OLD
        # with torch.no_grad():
        #     mel = mel.to(self.device)
        #     audio = self.vocoder(mel).clamp(-1, 1)
        #     if denoiser_strength > 0 and self.denoiser is not None:
        #         audio = self.denoiser(audio.squeeze(), strength=denoiser_strength)
        #     return audio.cpu().squeeze()

    def to(self, device: str):
        """Move all components to specified device."""
        self.device = device
        return self

