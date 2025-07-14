import torch
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN


class VocoderService:
    def __init__(self, vocoder_name: str, checkpoint_path: str, device: torch.device):
        self.vocoder_name = vocoder_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.vocoder = None
        self.denoiser = None

        self._load_vocoder()

    def _load_vocoder(self):
        print(f"[VocoderService] Loading vocoder '{self.vocoder_name}'...")
        if self.vocoder_name in ("hifigan_T2_v1", "hifigan_univ_v1"):
            self.vocoder = self._load_hifigan()
        else:
            raise NotImplementedError(f"Vocoder '{self.vocoder_name}' not supported.")

        self.denoiser = Denoiser(self.vocoder, mode="zeros")
        print(f"[VocoderService] Vocoder '{self.vocoder_name}' loaded.")

    def _load_hifigan(self):
        h = AttrDict(v1)
        model = HiFiGAN(h).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)["generator"]
        model.load_state_dict(state_dict)
        model.eval()
        model.remove_weight_norm()
        return model

    def vocoder_infer(self, mel, denoiser_strength=0.00025):
        """
        Convert mel spectrogram to waveform using vocoder and denoiser.
        """
        with torch.no_grad():
            audio = self.vocoder(mel).clamp(-1, 1)
            if self.denoiser is not None and denoiser_strength:
                audio = self.denoiser(audio.squeeze(), strength=denoiser_strength).cpu().squeeze()
            return audio.cpu().squeeze()

    def get_vocoder(self):
        return self.vocoder

    def get_denoiser(self):
        return self.denoiser
