"""Pocket TTS — Kyutai Labs small, fast TTS with voice cloning via kvcache.

Install: pip install livekit-streaming-tts[pocket]
Repo:    https://github.com/kyutai-labs/pocket-tts
Voices:  https://huggingface.co/kyutai/pocket-tts (preset names like "alba")
         and any local .wav, hf:// URL, or exported .safetensors kvcache.

Why a voice is fast after the first use:
    Encoding a wav into a model state (KV cache) is the expensive step.
    The state is kept in memory and passed to generate_audio with
    copy_state=True, so later utterances reuse that object and the library
    copies it instead of encoding again. The first encode is also written
    to a .safetensors file. The next process loads that tensor.
    A preset on the bundled Kyutai model loads Kyutai's embedding rather
    than a wav. pocket-tts does not load a .pt voice file.

Community weights:
    POCKET_TTS_MODEL=owner/repo
    POCKET_TTS_CONFIG=/path/to/config.yaml   (or hf://owner/repo/config.yaml)
    Leave both unset for Kyutai's bundled English model. pocket-tts
    load_model() wants a local yaml, and community configs usually name
    model.safetensors and tokenizer.model relative to that yaml, so we
    download the snapshot and rewrite those paths before loading.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import AudioChunk, EngineCapabilities, GenerationParams, TTSEngine
from .pocket_voice import VoiceStateCache

logger = logging.getLogger("livekit_streaming_tts.pocket")

# Paths inside a Pocket yaml that are files on disk (or hf:// / http URLs).
_ASSET_KEYS = (
    ("weights_path",),
    ("weights_path_without_voice_cloning",),
    ("flow_lm", "weights_path"),
    ("flow_lm", "lookup_table", "tokenizer_path"),
    ("mimi", "weights_path"),
)


def _get_in(data: dict, keys: tuple[str, ...]):
    node = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_in(data: dict, keys: tuple[str, ...], value: str) -> None:
    node = data
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


def _is_relative_asset(value: str) -> bool:
    if value.startswith(("hf://", "http://", "https://")):
        return False
    return not os.path.isabs(value)


def _parse_hf_uri(uri: str) -> tuple[str, str, Optional[str]]:
    rest = uri.removeprefix("hf://")
    revision = None
    if "@" in rest:
        rest, revision = rest.rsplit("@", 1)
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"Expected hf://owner/repo/config.yaml, got {uri!r}")
    return "/".join(parts[:2]), "/".join(parts[2:]), revision


def _download_config(repo_id: str, filename: str, revision: Optional[str]) -> Path:
    """Fetch a community repo and return the local config.yaml path.

    Only the config and the relative weight/tokenizer files it names are
    downloaded. hf:// and http weight URLs are left for pocket-tts.
    """

    import yaml
    from huggingface_hub import hf_hub_download, snapshot_download

    cfg_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    data = yaml.safe_load(Path(cfg_file).read_text())
    patterns = {filename}
    if isinstance(data, dict):
        for keys in _ASSET_KEYS:
            value = _get_in(data, keys)
            if isinstance(value, str) and _is_relative_asset(value):
                patterns.add(value)
    snapshot = Path(
        snapshot_download(
            repo_id,
            revision=revision,
            allow_patterns=sorted(patterns),
        )
    )
    config_path = snapshot / filename
    if not config_path.is_file():
        raise FileNotFoundError(f"{filename} not found in {repo_id}")
    return config_path


def _locate_config(spec: str) -> Path:
    if spec.startswith("hf://"):
        repo_id, filename, revision = _parse_hf_uri(spec)
        return _download_config(repo_id, filename, revision)

    expanded = os.path.abspath(os.path.expanduser(spec))
    if os.path.isdir(expanded):
        candidate = os.path.join(expanded, "config.yaml")
        if os.path.isfile(candidate):
            return Path(candidate)
        raise FileNotFoundError(f"No config.yaml in {expanded}")
    if os.path.isfile(expanded):
        return Path(expanded)
    # Hugging Face repo id: owner/name. Not a local path.
    if spec.count("/") == 1 and not spec.endswith((".yaml", ".yml")):
        return _download_config(spec, "config.yaml", None)
    raise FileNotFoundError(
        f"Pocket config not found: {spec}. Set POCKET_TTS_MODEL to an "
        "owner/name repo id, or POCKET_TTS_CONFIG to a local .yaml "
        "(or hf://owner/repo/config.yaml)."
    )


def _materialize_config(config_path: Path) -> str:
    """Return a yaml whose weight and tokenizer paths are absolute.

    pocket-tts opens relative paths against the process cwd, not the
    directory that holds the yaml. Community cards ship
    `weights_path: model.safetensors` and `tokenizer_path: tokenizer.model`.
    """

    import yaml

    from .._paths import data_dir

    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Pocket config is not a mapping: {config_path}")

    root = str(config_path.parent)
    changed = False
    for keys in _ASSET_KEYS:
        value = _get_in(data, keys)
        if isinstance(value, str) and _is_relative_asset(value):
            _set_in(data, keys, os.path.abspath(os.path.join(root, value)))
            changed = True
    if not changed:
        return str(config_path)

    payload = yaml.safe_dump(data, sort_keys=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    dest_dir = Path(data_dir("pocket-configs"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{digest}.yaml"
    dest.write_text(payload)
    return str(dest)


def resolve_pocket_config() -> Optional[str]:
    """Local yaml for TTSModel.load_model, or None for the Kyutai default.

    POCKET_TTS_CONFIG wins as a local file, a directory containing
    config.yaml, an hf://owner/repo/file.yaml URI, or an owner/name repo id.
    POCKET_TTS_MODEL is the same, intended for a repo id. Set only one.
    """

    config_env = os.getenv("POCKET_TTS_CONFIG", "").strip()
    model_env = os.getenv("POCKET_TTS_MODEL", "").strip()
    if config_env and model_env:
        raise ValueError("Set only one of POCKET_TTS_CONFIG or POCKET_TTS_MODEL.")
    spec = config_env or model_env
    if not spec:
        return None
    return _materialize_config(_locate_config(spec))


class PocketEngine(TTSEngine):
    name = "pocket"
    native_sample_rate = 24000  # overridden in __init__ from the model
    capabilities = EngineCapabilities(
        voice_cloning=True,
        streaming_generation=False,
        multilingual=False,
        speed_control=False,
    )

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        prewarm_voices: Optional[list[str]] = None,
    ) -> None:
        from pocket_tts import TTSModel

        # Unset env keeps Kyutai English (TTSModel.load_model() with no args).
        # A community repo id is not a load_model() argument in pocket-tts 2.x.
        config_path = resolve_pocket_config()
        self._custom_weights = config_path is not None
        if config_path:
            self._model = TTSModel.load_model(config=config_path)
        else:
            self._model = TTSModel.load_model()
        # The model exposes its native rate after load.
        self.native_sample_rate = int(self._model.sample_rate)
        # In-memory voice states, plus a safetensors export so a restart
        # loads the tensor instead of encoding the wav again.
        # POCKET_TTS_CACHE_DIR overrides the default user data dir.
        from .._paths import data_dir
        cache_root = cache_dir or data_dir(
            "pocket-cache", env_var="POCKET_TTS_CACHE_DIR"
        )
        try:
            os.makedirs(cache_root, exist_ok=True)
        except (PermissionError, OSError):
            # In-memory state still works for this process.
            cache_root = ""
        self._voices = VoiceStateCache(
            self._model, cache_root, custom_weights=self._custom_weights
        )

        # Warm the default voice so the first turn does not encode it.
        # POCKET_TTS_PREWARM=alba,ex01 tts-server ...
        env_warm = os.getenv("POCKET_TTS_PREWARM", "")
        warm_list = (
            prewarm_voices
            if prewarm_voices is not None
            else ([v.strip() for v in env_warm.split(",") if v.strip()] or ["alba"])
        )
        for voice_name in warm_list:
            try:
                self._voices.get(voice_name)
            except Exception:
                logger.debug("skipping pocket prewarm voice %s", voice_name, exc_info=True)

    def register_voice(
        self, voice_id: str, audio_path: str, ref_text: Optional[str] = None
    ) -> None:
        # First call encodes audio_path (or loads a tensor). Later calls
        # reuse the state stored under voice_id. ref_text is unused.
        self._voices.get(voice_id, source=audio_path)

    def delete_voice(self, voice_id: str) -> None:
        self._voices.discard(voice_id)

    def list_voices(self) -> list[str]:
        return self._voices.keys()

    def generate(self, params: GenerationParams) -> AudioChunk:
        # Pocket TTS requires a voice state. Presets, wav paths, hf:// wavs,
        # and .safetensors tensors share one cache. `text` must already be a
        # spoken span: pause markers are removed before this runs, because
        # prepare_text_prompt would read them aloud.
        state = self._voices.state_for_generation(params.voice, params.ref_audio)
        audio = self._model.generate_audio(state, params.text, copy_state=True)
        # Pocket returns a 1-D torch tensor of PCM samples.
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        else:
            audio = np.asarray(audio)
        return AudioChunk(
            samples=audio.astype(np.float32),
            sample_rate=self.native_sample_rate,
            is_final=True,
        )
