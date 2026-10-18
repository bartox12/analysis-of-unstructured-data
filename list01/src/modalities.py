from __future__ import annotations
import shutil
from pathlib import Path
from typing import Any
import librosa
import numpy as np
import pandas as pd
import skimage.data
import skimage.io
from sklearn.datasets import fetch_20newsgroups

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TEXT_FILE = "text_sample.txt"
IMAGE_FILE = "image_sample.png"
AUDIO_FILE = "audio_sample.ogg"
TEXT_MIN_CHARS = 400
TEXT_MAX_CHARS = 1500


def _pick_text_document() -> str:
    """Return the first 20 Newsgroups ``sci.space`` post of moderate length.

    Returns
    -------
    str
        The first training document whose stripped length lies in
        ``[TEXT_MIN_CHARS, TEXT_MAX_CHARS]`` (headers, footers and quotes removed).

    Raises
    ------
    RuntimeError
        If no document satisfies the length criterion.
    """
    bunch = fetch_20newsgroups(
        subset="train",
        categories=["sci.space"],
        remove=("headers", "footers", "quotes"),
    )
    for doc in bunch.data:
        doc = doc.strip()
        if TEXT_MIN_CHARS <= len(doc) <= TEXT_MAX_CHARS:
            return doc
    raise RuntimeError(
        f"No sci.space document with {TEXT_MIN_CHARS}-{TEXT_MAX_CHARS} characters."
    )


def fetch_samples(data_dir: Path = DATA_DIR, overwrite: bool = False) -> dict[str, Path]:
    """Write one text, one image and one audio sample into ``data_dir``.

    Each file is written only if it does not exist yet (or ``overwrite`` is set), so
    after the first call everything is read from local files. The first call needs
    network access (scikit-learn and librosa download and cache their datasets).

    Parameters
    ----------
    data_dir : Path, default DATA_DIR
        Directory in which the samples are stored.
    overwrite : bool, default False
        Re-create files that already exist.

    Returns
    -------
    dict[str, Path]
        Paths keyed by modality: ``"text"``, ``"image"`` and ``"audio"``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "text": data_dir / TEXT_FILE,
        "image": data_dir / IMAGE_FILE,
        "audio": data_dir / AUDIO_FILE,
    }

    if overwrite or not paths["text"].exists():
        paths["text"].write_text(_pick_text_document(), encoding="utf-8", newline="\n")
    if overwrite or not paths["image"].exists():
        skimage.io.imsave(paths["image"], skimage.data.chelsea())
    if overwrite or not paths["audio"].exists():
        # Byte copy of the original Ogg Vorbis file: no decoding / re-encoding.
        shutil.copyfile(librosa.ex("trumpet"), paths["audio"])

    return paths


def load_text(path: Path) -> np.ndarray:
    """Load a UTF-8 text file as a 1-D array of its UTF-8 bytes.

    Parameters
    ----------
    path : Path
        Path to a UTF-8 encoded text file.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bytes,)`` and dtype ``uint8``.
    """
    text = Path(path).read_text(encoding="utf-8")
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def decode_text(arr: np.ndarray) -> str:
    """Invert :func:`load_text`: decode an array of UTF-8 bytes back to a string.

    Parameters
    ----------
    arr : np.ndarray
        1-D ``uint8`` array of UTF-8 bytes.

    Returns
    -------
    str
        The decoded text.
    """
    return arr.tobytes().decode("utf-8")


def load_image(path: Path) -> np.ndarray:
    """Load an image in its native layout, without any normalisation.

    Parameters
    ----------
    path : Path
        Path to an image file.

    Returns
    -------
    np.ndarray
        Array of shape ``(H, W, 3)`` and dtype ``uint8`` for an RGB image.
    """
    return skimage.io.imread(path)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio clip at its native sampling rate and channel count.

    librosa decodes the samples to ``float32`` in ``[-1, 1]``.

    Parameters
    ----------
    path : Path
        Path to an audio file.

    Returns
    -------
    waveform : np.ndarray
        Array of shape ``(n_samples,)`` for mono or ``(n_channels, n_samples)``
        otherwise, dtype ``float32``.
    sr : int
        Sampling rate in Hz.
    """
    waveform, sr = librosa.load(path, sr=None, mono=False)
    return waveform, int(sr)


def describe_array(name: str, arr: np.ndarray, **extra: Any) -> dict[str, Any]:
    """Summarise an array's layout and value range.

    Parameters
    ----------
    name : str
        Modality name stored in the ``"modality"`` field.
    arr : np.ndarray
        The array to describe.
    **extra
        Additional modality-specific fields (e.g. ``sr`` or ``n_chars``).

    Returns
    -------
    dict[str, Any]
        modality, shape, ndim, dtype, size, nbytes, min, max and the extras.
    """
    return {
        "modality": name,
        "shape": tuple(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "size": arr.size,
        "nbytes": arr.nbytes,
        "min": arr.min().item(),
        "max": arr.max().item(),
        **extra,
    }


def load_all(data_dir: Path = DATA_DIR) -> dict[str, np.ndarray]:
    """Load the three samples written by :func:`fetch_samples`.

    Parameters
    ----------
    data_dir : Path, default DATA_DIR
        Directory containing the samples.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays keyed by modality: ``"text"``, ``"image"`` and ``"audio"``.
    """
    data_dir = Path(data_dir)
    waveform, _ = load_audio(data_dir / AUDIO_FILE)
    return {
        "text": load_text(data_dir / TEXT_FILE),
        "image": load_image(data_dir / IMAGE_FILE),
        "audio": waveform,
    }


def summary_table(arrays: dict[str, np.ndarray], sr: int) -> pd.DataFrame:
    """Build a one-row-per-modality summary of the loaded arrays.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Arrays keyed by ``"text"``, ``"image"`` and ``"audio"``.
    sr : int
        Sampling rate of the audio array in Hz.

    Returns
    -------
    pd.DataFrame
        Output of :func:`describe_array` per modality, indexed by modality.
    """
    text, image, audio = arrays["text"], arrays["image"], arrays["audio"]
    rows = [
        describe_array("text", text, n_chars=len(decode_text(text))),
        describe_array("image", image),
        describe_array("audio", audio, sr=sr, duration_s=audio.shape[-1] / sr),
    ]
    table = pd.DataFrame(rows).set_index("modality")
    # Integer extras exist only for one modality; keep them integer despite the NaNs.
    return table.astype({"n_chars": "Int64", "sr": "Int64"})
