"""Quantize SmolVLM-256M.

- Mac (MPS/MLX): converts to MLX 4-bit format for fast local inference
- Linux/CUDA:    NF4 via bitsandbytes for submission-ready weights

MLX quantized model goes to models/smolvlm-256m-mlx/
CUDA quantized model goes to models/smolvlm-256m-4bit/
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent.parent
MODEL_IN = ROOT / "models" / "smolvlm-256m"
MODEL_OUT_MLX = ROOT / "models" / "smolvlm-256m-mlx"
MODEL_OUT_4BIT = ROOT / "models" / "smolvlm-256m-4bit"


def size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def quantize_mlx():
    """4-bit quantization via mlx-vlm — runs on Apple Silicon GPU."""
    from mlx_vlm.utils import load, save_weights
    from mlx_vlm.convert import convert as vlm_convert

    if MODEL_OUT_MLX.exists():
        print(f"MLX model already exists at {MODEL_OUT_MLX} ({size_mb(MODEL_OUT_MLX):.1f} MB)")
        return

    print(f"Converting {MODEL_IN} to MLX 4-bit...")
    vlm_convert(
        str(MODEL_IN),
        mlx_path=str(MODEL_OUT_MLX),
        quantize=True,
        q_bits=4,
    )
    # Strip onnx directory — copied verbatim but not needed for inference
    import shutil
    onnx_dir = MODEL_OUT_MLX / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    print(f"MLX 4-bit model saved: {size_mb(MODEL_OUT_MLX):.1f} MB")


def quantize_cuda():
    """NF4 quantization via bitsandbytes — requires CUDA. Use on Kaggle/Colab."""
    from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq

    if MODEL_OUT_4BIT.exists():
        print(f"4-bit model already exists at {MODEL_OUT_4BIT} ({size_mb(MODEL_OUT_4BIT):.1f} MB)")
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print(f"Loading {MODEL_IN} with NF4 quantization...")
    model = AutoModelForVision2Seq.from_pretrained(
        str(MODEL_IN),
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_IN))

    MODEL_OUT_4BIT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUT_4BIT))
    processor.save_pretrained(str(MODEL_OUT_4BIT))
    print(f"NF4 model saved: {size_mb(MODEL_OUT_4BIT):.1f} MB")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA detected — running NF4 bitsandbytes quantization.")
        quantize_cuda()
    elif torch.backends.mps.is_available():
        print("Apple Silicon detected — running MLX 4-bit quantization.")
        quantize_mlx()
    else:
        print("CPU only — running MLX quantization (will be slow).")
        quantize_mlx()
