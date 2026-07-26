import sys
import hashlib
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DeepFaceLab.setting import INSIGHTFACE_MODEL_DIR, INSIGHTFACE_DIR


MODEL_URLS = {
    "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
    "inswapper_128.onnx": "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx",
}

MODEL_SHA1 = {}


def download_file(url: str, output_path: Path) -> None:
    import urllib.request
    print(f"Downloading {url} -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(output_path))
    print(f"Downloaded {output_path.name}")


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    import zipfile
    print(f"Extracting {zip_path} -> {output_dir}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(output_dir))
    print(f"Extracted to {output_dir}")


def download_insightface_models() -> None:
    model_dir = INSIGHTFACE_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    antelopev2_dir = model_dir / "antelopev2"
    if antelopev2_dir.exists() and any(antelopev2_dir.iterdir()):
        print("antelopev2 models already exist, skipping.")
    else:
        zip_path = model_dir / "antelopev2.zip"
        if not zip_path.exists():
            download_file(MODEL_URLS["antelopev2"], zip_path)
        extract_zip(zip_path, model_dir)
        zip_path.unlink(missing_ok=True)

    inswapper_path = model_dir / "inswapper_128.onnx"
    if inswapper_path.exists():
        print("inswapper_128.onnx already exists, skipping.")
    else:
        download_file(MODEL_URLS["inswapper_128.onnx"], inswapper_path)

    print("All insightface models downloaded.")


if __name__ == "__main__":
    download_insightface_models()
