"""
Alzheimer's MRI classification pipeline.

Upload -> validate -> RGB -> resize 224x224 -> ImageNet normalize
       -> model.pkl (DenseNet-121) -> 4-class softmax -> prediction + probabilities
"""
from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

# --------------------------------------------------------------------------
# CONFIG  -- CLASS_NAMES order MUST match the label order used in training.
# (ImageFolder sorts folders alphabetically, which gives the order below.)
# --------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_FILE_MB = 20
MIN_SIDE_PX = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageInput = Union[str, Path, bytes, io.BytesIO, "object"]  # path, bytes, or file-like


class InvalidImageError(ValueError):
    """Raised when the uploaded file is not a usable image."""


# --------------------------------------------------------------------------
# 1. IMAGE VALIDATION  +  2. RGB CONVERSION
# --------------------------------------------------------------------------
def load_and_validate(source: ImageInput, filename: str | None = None) -> Image.Image:
    """Validate the upload and return an RGB PIL image."""
    # Read raw bytes
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise InvalidImageError(f"File not found: {path}")
        filename = filename or path.name
        data = path.read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    elif hasattr(source, "read"):  # file-like (e.g. Streamlit UploadedFile)
        filename = filename or getattr(source, "name", None)
        data = source.read()
    else:
        raise InvalidImageError("Unsupported input type.")

    if filename and Path(filename).suffix.lower() not in ALLOWED_EXT:
        raise InvalidImageError(
            f"Unsupported file type '{Path(filename).suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXT))}"
        )
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise InvalidImageError(f"File is larger than {MAX_FILE_MB} MB.")

    # Verify it's a real, uncorrupted image (verify() invalidates the object,
    # so we open it a second time to actually use it).
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as e:
        raise InvalidImageError(f"Not a valid or readable image: {e}") from e

    if min(img.size) < MIN_SIDE_PX:
        raise InvalidImageError(f"Image too small ({img.size[0]}x{img.size[1]}px).")

    # Handle 16-bit / float grayscale TIFFs before RGB conversion
    if img.mode in ("I;16", "I", "F"):
        import numpy as np

        arr = np.asarray(img, dtype="float32")
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8) * 255
        img = Image.fromarray(arr.astype("uint8"))

    # RGB conversion (MRI is typically grayscale; replicates to 3 channels)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))  # black bg for MRI
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# --------------------------------------------------------------------------
# 3. RESIZE 224x224  +  4. IMAGENET NORMALIZATION
# --------------------------------------------------------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # HWC uint8 [0,255] -> CHW float [0,1]
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# --------------------------------------------------------------------------
# 5. LOAD model.pkl  (DenseNet-121)
# --------------------------------------------------------------------------
def load_model(path: Union[str, Path] = MODEL_PATH) -> nn.Module:
    """
    Loads the saved DenseNet-121. Handles both:
      * a fully pickled model   (torch.save(model, ...))   <- your file
      * a state_dict            (torch.save(model.state_dict(), ...))

    SECURITY: full-model pickles can execute code on load. Only load files you trust.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    obj = torch.load(path, map_location=DEVICE, weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        state = obj.get("state_dict") or obj.get("model_state_dict") or obj
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
        model.load_state_dict(state)
    else:
        raise TypeError(f"Unexpected object in model file: {type(obj)}")

    if isinstance(model, nn.DataParallel):
        model = model.module

    model.to(DEVICE).eval()

    # Sanity check: output size must match number of class names
    out_features = _output_size(model)
    if out_features is not None and out_features != len(CLASS_NAMES):
        raise ValueError(
            f"Model outputs {out_features} classes but CLASS_NAMES has {len(CLASS_NAMES)}."
        )
    return model


def _output_size(model: nn.Module):
    clf = getattr(model, "classifier", None)
    if isinstance(clf, nn.Linear):
        return clf.out_features
    if isinstance(clf, nn.Sequential):
        for layer in reversed(clf):
            if isinstance(layer, nn.Linear):
                return layer.out_features
    return None


# --------------------------------------------------------------------------
# 6. INFERENCE: DenseNet-121 -> 4-class softmax -> prediction + confidence
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    label: str
    confidence: float                 # 0-1
    probabilities: Dict[str, float]   # every class, sums to 1

    def __str__(self) -> str:
        lines = [f"Prediction : {self.label}", f"Confidence : {self.confidence:.2%}", "Probabilities:"]
        for k, v in sorted(self.probabilities.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<18} {v:7.2%}")
        return "\n".join(lines)


@torch.inference_mode()
def predict(model: nn.Module, source: ImageInput, filename: str | None = None) -> Prediction:
    img = load_and_validate(source, filename)             # validation + RGB
    x = preprocess(img).unsqueeze(0).to(DEVICE)           # resize + normalize + batch dim
    logits = model(x)                                     # DenseNet-121 forward
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()  # 4-class softmax

    idx = max(range(len(probs)), key=probs.__getitem__)
    return Prediction(
        label=CLASS_NAMES[idx],
        confidence=probs[idx],
        probabilities=dict(zip(CLASS_NAMES, probs)),
    )


# --------------------------------------------------------------------------
# CLI:  python pipeline.py path/to/mri.jpg
# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python pipeline.py <mri_image>")
    try:
        print(predict(load_model(), sys.argv[1]))
    except InvalidImageError as e:
        sys.exit(f"Invalid image: {e}")
