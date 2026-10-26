"""In-process Pocket voice states.

Encoding a reference wav (``get_state_for_audio_prompt`` on audio) is the
expensive step. A ``.safetensors`` voice state is a load, not an encode.
This cache keeps the state object for the life of the process and writes
an export so the next process loads the tensor instead of encoding again.

``generate_audio(..., copy_state=True)`` deep-copies that object because
generation mutates the KV cache. Callers pass the cached object through
and let the library copy it. A second copy here would add latency and
is not required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading

logger = logging.getLogger("livekit_streaming_tts.pocket")

_SAFE_VOICE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def voice_cache_filename(voice: str) -> str:
    """Filesystem-safe export name. Stable for a given voice string."""

    digest = hashlib.sha256(voice.encode("utf-8")).hexdigest()[:20]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", voice).strip("._-")[:40] or "voice"
    return f"{slug}-{digest}.safetensors"


def _strip_hf_revision(spec: str) -> str:
    path = spec.split("?", 1)[0].strip()
    if path.startswith("hf://") and "@" in path:
        path = path.rsplit("@", 1)[0]
    return path


def is_voice_tensor(spec: str) -> bool:
    """True when pocket-tts will import a safetensors voice state, not audio."""

    return _strip_hf_revision(spec).lower().endswith(".safetensors")


def is_torch_pickle(spec: str) -> bool:
    """``.pt`` / ``.pth`` voice files. pocket-tts 2.1.0 cannot load them."""

    suffix = _strip_hf_revision(spec).lower()
    return suffix.endswith(".pt") or suffix.endswith(".pth")


def resolve_voice_prompt(prompt: str, *, custom_weights: bool) -> str:
    """Prompt string passed to ``get_state_for_audio_prompt``.

    Bundled Kyutai weights accept a preset name (``alba``) and load a
    precomputed embedding. A community yaml is not one of those configs,
    so the same name is cloned from the preset's wav URL. Local files,
    ``hf://`` URLs, and http URLs are passed through.
    """

    if not custom_weights:
        return prompt
    if prompt.startswith(("hf://", "http://", "https://")):
        return prompt
    if os.path.isabs(prompt) or prompt.endswith(
        (".wav", ".mp3", ".flac", ".ogg", ".safetensors", ".pt", ".pth")
    ):
        return prompt
    try:
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
    except Exception:
        return prompt
    return _ORIGINS_OF_PREDEFINED_VOICES.get(prompt, prompt)


def _preset_needs_audio(name: str) -> bool:
    """True when ``name`` is a Kyutai preset id, not a file or URL.

    The precomputed embedding for that id belongs to Kyutai's weights.
    Feeding it to a community model leaves generation out of distribution:
    it runs to the length cap and never emits EOS.
    """

    if not name or name.startswith(("hf://", "http://", "https://")):
        return False
    if os.path.isabs(name) or name.endswith(
        (".wav", ".mp3", ".flac", ".ogg", ".safetensors", ".pt", ".pth")
    ):
        return False
    return resolve_voice_prompt(name, custom_weights=True) != name


def _file_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _local_mtime(path: str) -> float | None:
    if path.startswith(("hf://", "http://", "https://")):
        return None
    try:
        if os.path.isfile(path):
            return os.path.getmtime(path)
    except OSError:
        return None
    return None


def _is_newer(mtime: float | None, seen: float | None) -> bool:
    if mtime is None or seen is None:
        return False
    return mtime > seen


def _export(state: object, dest: str) -> None:
    """Write a voice state atomically. pocket-tts reads this back via safetensors."""

    from pocket_tts import export_model_state

    directory = os.path.dirname(dest)
    if directory:
        os.makedirs(directory, exist_ok=True)
    partial = dest + ".partial"
    try:
        export_model_state(state, partial)
        os.replace(partial, dest)
    except Exception:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise


class VoiceStateCache:
    """Voice id or path → one model state object, plus a safetensors export."""

    def __init__(self, model: object, cache_dir: str, *, custom_weights: bool) -> None:
        self._model = model
        self._cache_dir = cache_dir
        self._custom_weights = custom_weights
        self._memory: dict[str, object] = {}
        self._origin: dict[str, str] = {}
        self._mtime: dict[str, float | None] = {}
        self._lock = threading.Lock()

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._memory)

    def contains(self, voice: str) -> bool:
        key = voice.strip()
        if not key:
            return False
        with self._lock:
            if key in self._memory or self._find_disk(key) is not None:
                return True
            return self._read_origin(key) is not None

    def discard(self, voice: str) -> None:
        """Drop memory and the exported tensor. Used when a voice is deleted."""

        key = voice.strip()
        if not key:
            return
        with self._lock:
            self._memory.pop(key, None)
            self._origin.pop(key, None)
            self._mtime.pop(key, None)
            for path in self._disk_candidates(key):
                try:
                    os.remove(path)
                except OSError:
                    pass
            meta = self._meta_path(key)
            if meta:
                try:
                    os.remove(meta)
                except OSError:
                    pass

    def state_for_generation(self, voice: str | None, ref_audio: str | None) -> object:
        """State for one ``generate`` call.

        A voice id that is already cached wins. Otherwise a reference path
        is cloned once and cached under that path. Preset names, wav paths,
        ``hf://`` wavs, and safetensors paths all go through :meth:`get`.
        """

        name = (voice or "").strip()
        ref = (ref_audio or "").strip()
        if name and self.contains(name):
            return self.get(name)
        if ref:
            return self.get(ref, source=ref)
        return self.get(name or "alba")

    def get(self, voice: str, *, source: str | None = None) -> object:
        """Return the cached state, loading a tensor or encoding audio once.

        ``source`` is the wav / URL / tensor to encode when ``voice`` is only
        an id (voice upload). Later ``get(voice)`` calls reuse the same object.
        """

        key = voice.strip()
        if not key:
            raise ValueError("voice is empty")
        explicit = source is not None and bool(source.strip())
        requested = resolve_voice_prompt(
            source.strip() if explicit else key,
            custom_weights=self._custom_weights,
        )
        if is_torch_pickle(requested):
            raise ValueError(
                "Pocket TTS loads a .safetensors voice state, not a .pt file. "
                "The server writes a .safetensors export the first time it "
                "encodes a voice; pass that path as voice."
            )

        with self._lock:
            if key in self._memory:
                origin = self._origin[key]
                candidate = requested if explicit else origin
                if candidate == origin and not _is_newer(
                    _local_mtime(candidate), self._mtime.get(key)
                ):
                    logger.debug("reusing voice state %s", key)
                    return self._memory[key]
                requested = candidate

            disk = self._find_disk(key)
            saved = self._read_origin(key)
            prompt = requested
            if disk is not None and self._can_load_disk(disk, key, prompt, saved, explicit):
                logger.info("loading cached voice tensor for %s", key)
                state = self._model.get_state_for_audio_prompt(disk)
                self._remember(key, state, saved or prompt)
                return state

            # A voice id has no wav of its own. If the saved source is a
            # file, encode that file, not the id string. A bare preset name
            # ("alba") is a Kyutai embedding; community weights must keep
            # the wav URL already resolved into `prompt`.
            if saved and not explicit and saved != prompt:
                if not (self._custom_weights and _preset_needs_audio(saved)):
                    prompt = saved

            if is_voice_tensor(prompt):
                logger.info("loading voice tensor %s", prompt)
                state = self._model.get_state_for_audio_prompt(prompt)
                self._write_origin(key, prompt)
                self._remember(key, state, prompt)
                return state

            logger.info("encoding voice %s", key)
            state = self._model.get_state_for_audio_prompt(prompt, truncate=True)
            self._write_disk(key, state, prompt)
            self._remember(key, state, prompt)
            return state

    def _remember(self, key: str, state: object, origin: str) -> None:
        self._memory[key] = state
        self._origin[key] = origin
        self._mtime[key] = _local_mtime(origin)

    def _can_load_disk(
        self,
        disk: str,
        key: str,
        requested: str,
        saved: str | None,
        explicit: bool,
    ) -> bool:
        """True when the export was produced for this prompt and is not stale."""

        # A previous Kyutai run exports preset "alba" as a precomputed
        # embedding. That file must not be reused once community weights
        # are loaded: pocket-tts then never emits EOS. A voice registered
        # under the same id still loads, because its origin is a wav path.
        if (
            self._custom_weights
            and not explicit
            and _preset_needs_audio(saved if saved is not None else key)
            and (saved is None or _preset_needs_audio(saved))
            and saved != requested
        ):
            logger.info(
                "ignoring Kyutai preset cache for %s; encoding the voice wav",
                key,
            )
            return False
        if saved is None:
            # Legacy `{name}.safetensors`, or an export written for the key itself.
            if explicit and requested != key:
                return False
            return not self._disk_stale(disk, _local_mtime(requested))
        if explicit and requested != saved:
            return False
        if _is_newer(_local_mtime(saved), _file_mtime(disk)):
            return False
        return True

    def _write_disk(self, voice: str, state: object, origin: str) -> None:
        if not self._cache_dir:
            return
        dest = os.path.join(self._cache_dir, voice_cache_filename(voice))
        try:
            _export(state, dest)
        except Exception:
            logger.warning("could not export voice tensor for %s", voice, exc_info=True)
            return
        self._write_origin(voice, origin)

    def _meta_path(self, voice: str) -> str | None:
        if not self._cache_dir:
            return None
        name = voice_cache_filename(voice)
        stem = name[: -len(".safetensors")]
        return os.path.join(self._cache_dir, stem + ".origin.json")

    def _write_origin(self, voice: str, origin: str) -> None:
        path = self._meta_path(voice)
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"origin": origin}, handle)
        except OSError:
            logger.warning("could not record voice origin for %s", voice, exc_info=True)

    def _read_origin(self, voice: str) -> str | None:
        path = self._meta_path(voice)
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return None
        origin = data.get("origin") if isinstance(data, dict) else None
        if isinstance(origin, str) and origin:
            return origin
        return None

    def _disk_stale(self, disk: str, source_mtime: float | None) -> bool:
        if source_mtime is None:
            return False
        try:
            return source_mtime > os.path.getmtime(disk)
        except OSError:
            return True

    def _disk_candidates(self, voice: str) -> list[str]:
        if not self._cache_dir:
            return []
        paths = [os.path.join(self._cache_dir, voice_cache_filename(voice))]
        if _SAFE_VOICE_ID.fullmatch(voice):
            paths.append(os.path.join(self._cache_dir, f"{voice}.safetensors"))
        return paths

    def _find_disk(self, voice: str) -> str | None:
        for path in self._disk_candidates(voice):
            if os.path.isfile(path):
                return path
        return None
