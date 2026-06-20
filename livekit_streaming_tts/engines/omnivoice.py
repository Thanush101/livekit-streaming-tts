"""OmniVoice — k2-fsa diffusion TTS with voice cloning.

Install: pip install livekit-streaming-tts[omnivoice]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine


class OmniVoiceEngine(TTSEngine):
    name = "omnivoice"
    native_sample_rate = 24000
    capabilities = EngineCapabilities(
        voice_cloning=True,
        streaming_generation=False,
        multilingual=True,
        speed_control=True,
    )

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        device: str = "cuda:0",
        dtype: str = "float16",
    ) -> None:
        from omnivoice import OmniVoice

        torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
        self._model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=torch_dtype)
        self._voice_cache: dict[str, object] = {}

    def register_voice(self, voice_id: str, audio_path: str, ref_text: Optional[str] = None) -> None:
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=audio_path, ref_text=ref_text,
        )
        self._voice_cache[voice_id] = prompt

    def delete_voice(self, voice_id: str) -> None:
        self._voice_cache.pop(voice_id, None)

    def list_voices(self) -> list[str]:
        return list(self._voice_cache.keys())

    def generate(self, params: GenerationParams) -> AudioChunk:
        from dataclasses import fields as dataclass_fields
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

        if params.seed is not None:
            torch.manual_seed(params.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(params.seed)

        # Pull engine-specific overrides from `extra`. The full set of
        # OmniVoice diffusion knobs lives here.
        all_cfg = {
            "num_step": params.extra.get("steps", 8),
            "guidance_scale": params.extra.get("guidance_scale", 2.0),
            "denoise": params.extra.get("denoise", True),
            "t_shift": params.extra.get("t_shift", 0.1),
            "class_temperature": params.extra.get("temperature", 0.0),
            "position_temperature": params.extra.get("position_temperature", 5.0),
            "layer_penalty_factor": params.extra.get("layer_penalty_factor", 5.0),
            "preprocess_prompt": params.extra.get("preprocess_prompt", True),
            "postprocess_output": params.extra.get("postprocess_output", False),
        }
        accepted = {f.name for f in dataclass_fields(OmniVoiceGenerationConfig)}
        gen_config = OmniVoiceGenerationConfig(**{k: v for k, v in all_cfg.items() if k in accepted})

        kwargs: dict[str, object] = {"text": params.text, "generation_config": gen_config}
        if params.extra.get("duration") is not None:
            kwargs["duration"] = params.extra["duration"]
        elif params.speed is not None:
            kwargs["speed"] = params.speed

        if params.voice and params.voice in self._voice_cache:
            kwargs["voice_clone_prompt"] = self._voice_cache[params.voice]
        elif params.ref_audio:
            kwargs["ref_audio"] = params.ref_audio
            if params.ref_text:
                kwargs["ref_text"] = params.ref_text
        if params.extra.get("instruct"):
            kwargs["instruct"] = params.extra["instruct"]

        out = self._model.generate(**kwargs)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, torch.Tensor):
            out = out.cpu().numpy()
        return AudioChunk(samples=out.astype(np.float32), sample_rate=self.native_sample_rate, is_final=True)
