"""
emotion_infer.py — loads a trained emotion model and predict the emotion for a single image.

Supported model formats (auto-detected by file extension or --backend):
  - TensorFlow / Keras SavedModel: .keras or .h5
  - TorchScript (NOT a raw .pth with state_dict): .pt (or .torchscript.pt)

Defaults assume FER2013 (48x48, grayscale) and 7 emotions:
['angry','disgust','fear','happy','sad','surprise','neutral']

USAGE
  As a module (used by provide_filename.py):
    from emotion_infer import process_file
    process_file("/path/to/image.jpg")  # prints and returns the label

  As a script:
    python emotion_infer.py /path/to/image.jpg --model model.keras
    python emotion_infer.py /path/to/image.jpg --model model.torchscript.pt --backend torch

Environment fallback:
  If --model is not given, it tries $EMOTION_MODEL, then common filenames:
  ./model.torchscript.pt
"""

from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

EMOTIONS_DEFAULT = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

MODEL_CANDIDATES = ["model.torchscript.pt", "model.pt"]

def _find_model_path(cli_path: Optional[str]) -> Path:
    if cli_path:
        p = Path(cli_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p
    env = os.getenv("EMOTION_MODEL")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    here = Path(__file__).resolve().parent
    for name in MODEL_CANDIDATES:
        for base in (here, Path.cwd()):
            p = (base / name).resolve()
            if p.exists():
                return p
    raise FileNotFoundError(
        "No model file provided. Pass --model, set $EMOTION_MODEL, or place one of "
        f"{MODEL_CANDIDATES} next to this script."
    )

def _infer_backend_from_path(model_path: Path, backend: Optional[str]) -> str:
    if backend:
        b = backend.lower().strip()
        if b in ("tf", "tensorflow", "keras"):
            return "tf"
        if b in ("torch", "pytorch"):
            return "torch"
        raise ValueError("--backend must be one of: tf, torch")
    ext = model_path.suffix.lower()
    if ext in (".keras", ".h5"):
        return "tf"
    if ext in (".pt",):
        return "torch"
    # fallback guess
    return "tf"

def _load_tf_model(path: Path):
    try:
        import tensorflow as tf  # type: ignore
    except Exception as e:
        raise RuntimeError("provide pt model") from e
    model = tf.keras.models.load_model(str(path), compile=False)
    return model, tf

def _load_torchscript_model(path: Path):
    try:
        import torch  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "install pytorch or provide keras model"
        ) from e
    try:
        with open(path, "rb") as f:
            model = torch.jit.load(f, map_location="cpu")
    except Exception:
        # Fallback to string path (works fine on pure ASCII paths)
        model = torch.jit.load(str(path), map_location="cpu")

    model.eval()
    return model, torch


def _read_image(
    image_path: Path,
    size: int,
    color: str
) -> np.ndarray:
    """
    Returns float32 array scaled to [0,1].
    color: 'grayscale' or 'rgb'
    """
    if color not in ("grayscale", "rgb"):
        raise ValueError("color must be 'grayscale' or 'rgb'")
    img = Image.open(image_path)
    img = img.convert("L" if color == "grayscale" else "RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0 
    return arr

def _tf_predict(
    tf, model, arr: np.ndarray, channels_last: bool = True, labels: List[str] = EMOTIONS_DEFAULT
) -> Tuple[str, float, np.ndarray]:
    try:
        in_shape = tuple(model.inputs[0].shape.as_list())  
    except Exception:
        in_shape = getattr(model, "input_shape", None)
        if isinstance(in_shape, (list, tuple)):
            in_shape = tuple(in_shape)
    # grayscale vs rgb
    if arr.ndim == 2:
        # grayscale
        x = arr[..., None]  
    else:
        x = arr 
    x = x[None, ...].astype(np.float32)  
    # If model expects channels_first, transpose
    if in_shape and len(in_shape) == 4:
        # in_shape looks like (None, H, W, C) or (None, C, H, W)
        if in_shape[1] in (1, 3) and in_shape[-1] not in (1, 3):
            # (None, C, H, W)
            x = np.transpose(x, (0, 3, 1, 2))
    preds = model.predict(x, verbose=0)
    if hasattr(tf.nn, "softmax"):
        probs = tf.nn.softmax(preds, axis=-1).numpy()
    else:
        # best-effort softmax
        e = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
        probs = e / np.sum(e, axis=-1, keepdims=True)

    p = probs[0]
    idx = int(np.argmax(p))
    label = labels[idx] if len(labels) > idx else f"class_{idx}"
    conf = float(p[idx])
    return label, conf, p

def _torch_predict(
    torch, model, arr: np.ndarray, color: str, size: int, labels: List[str] = EMOTIONS_DEFAULT
) -> Tuple[str, float, np.ndarray]:
    # TorchScript expects [N,C,H,W] float32
    if arr.ndim == 2:
        # grayscale
        x = arr[None, None, :, :]  
    else:
        # rgb
        x = np.transpose(arr, (2, 0, 1))[None, ...] 

    x_t = torch.from_numpy(x.astype(np.float32))

    if color == "grayscale":
       
        x_t = (x_t - 0.5) / 0.5
    else:
        pass
    with torch.no_grad():
        out = model(x_t)

    if isinstance(out, (tuple, list)):
        out = out[0]
    out_np = out.numpy()

    # softmax
    e = np.exp(out_np - np.max(out_np, axis=-1, keepdims=True))
    probs = e / np.sum(e, axis=-1, keepdims=True)
    p = probs[0]
    idx = int(np.argmax(p))
    label = labels[idx] if len(labels) > idx else f"class_{idx}"
    conf = float(p[idx])
    return label, conf, p

def predict_emotion(
    image_path: str,
    model_path: Optional[str] = None,
    backend: Optional[str] = None,
    image_size: int = 48,
    color: str = "grayscale",
    labels: Optional[List[str]] = None,
) -> Tuple[str, float, np.ndarray]:
    """
    Returns (label, confidence, probability_vector).
    """
    labels = labels or EMOTIONS_DEFAULT
    img_p = Path(image_path).expanduser()
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found: {img_p}")
    mdl_p = _find_model_path(model_path)
    be = _infer_backend_from_path(mdl_p, backend)

    arr = _read_image(img_p, image_size, color)

    if be == "tf":
        model, tf = _load_tf_model(mdl_p)
        label, conf, probs = _tf_predict(tf, model, arr, True, labels)
    else:
        model, torch = _load_torchscript_model(mdl_p)
        label, conf, probs = _torch_predict(torch, model, arr, color, image_size, labels)

    return label, conf, probs

# public API for provide_filename.py 

def process_file(path: str) -> str:
    """
    Called by provide_filename.py. Prints and returns the predicted emotion.
    Uses $EMOTION_MODEL or a local model file (see MODEL_CANDIDATES).
    """
    try:
        label, conf, _ = predict_emotion(path)
        print(f"Predicted emotion: {label}  (confidence: {conf:.3f})")
        return label
    except Exception as e:
        raise

# CLI 
def _parse_labels(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Predict emotion from an image file.")
    p.add_argument("image", help="Path to the input image")
    p.add_argument("--model", help="Path to model (.keras/.h5 or .pt). Defaults to $EMOTION_MODEL or common names.", default=None)
    p.add_argument("--backend", help="Force backend: tf or torch", choices=["tf","torch"])
    p.add_argument("--size", type=int, default=48, help="Image size (short side). Default: 48")
    p.add_argument("--color", choices=["grayscale","rgb"], default="grayscale", help="Color mode for pre-processing. Default: grayscale")
    p.add_argument("--labels", help="Comma-separated labels matching your model's classes.")
    args = p.parse_args(argv)

    labels = _parse_labels(args.labels) or EMOTIONS_DEFAULT
    label, conf, _ = predict_emotion(
        args.image,
        model_path=args.model,
        backend=args.backend,
        image_size=args.size,
        color=args.color,
        labels=labels,
    )
    print(f"{label}")

if __name__ == "__main__":
    main()
