"""
GGUF 部署 — 将 FusionModel 导出为 GGUF 格式（用于 llama.cpp）

使用方式:
    python deployment/export_gguf.py --checkpoint output/mini_model --output output/gguf/model.gguf
    python deployment/export_gguf.py --checkpoint output/mini_model --output output/gguf/model-Q4_K_M.gguf --quantize Q4_K_M

依赖: llama.cpp (自动检测)
"""
import sys
import os
import json
import argparse
import struct
from pathlib import Path

sys.path.insert(0, '.')


def find_llama_cpp():
    """Find llama.cpp installation."""
    candidates = [
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/src/llama.cpp"),
        "/usr/local/bin/llama-cli",
        "llama-cli",
    ]
    for c in candidates:
        if os.path.isfile(c) or os.path.isdir(c):
            return c
    # Check PATH
    import shutil
    if shutil.which("llama-cli") or shutil.which("llama-cpp"):
        return shutil.which("llama-cli") or shutil.which("llama-cpp")
    return None


def export_to_hf_format(model, output_dir):
    """Export model to HuggingFace format (safetensors + config.json).
    
    This is the canonical intermediate format. Use llama.cpp's convert
    script to produce GGUF from this.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model using HF's save_pretrained (produces safetensors)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    
    # Write tokenizer config for llama.cpp compatibility
    tokenizer_config = {
        "model_type": "fusion",
        "vocab_size": model.config.vocab_size,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    with open(output_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"[EXPORT] HF format saved to {output_dir}")
    return output_dir


def convert_to_gguf(hf_dir, output_path, llama_cpp_path=None, quantize=None):
    """Convert HuggingFace format to GGUF using llama.cpp's convert script.
    
    Args:
        hf_dir: Path to HF format directory
        output_path: Output GGUF file path
        llama_cpp_path: Path to llama.cpp installation
        quantize: Quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0)
    """
    import subprocess
    
    llama_cpp = llama_cpp_path or find_llama_cpp()
    
    if llama_cpp is None:
        print("[WARNING] llama.cpp not found. Saving HF format only.")
        print("  To convert to GGUF, install llama.cpp and run:")
        print(f"  python convert_hf_to_gguf.py {hf_dir} --outfile {output_path}")
        return False
    
    # Find convert script
    convert_script = None
    if os.path.isdir(llama_cpp):
        for name in ["convert_hf_to_gguf.py", "convert.py"]:
            candidate = os.path.join(llama_cpp, name)
            if os.path.isfile(candidate):
                convert_script = candidate
                break
    
    if convert_script is None:
        print("[WARNING] llama.cpp convert script not found. Saving HF format only.")
        print(f"  Manual conversion: python convert_hf_to_gguf.py {hf_dir} --outfile {output_path}")
        return False
    
    # Run conversion
    cmd = [sys.executable, convert_script, str(hf_dir), "--outfile", str(output_path)]
    print(f"[CONVERT] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Conversion failed: {result.stderr}")
        return False
    
    print(f"[CONVERT] GGUF saved to {output_path}")
    
    # Quantize if requested
    if quantize and os.path.isdir(llama_cpp):
        quantize_bin = os.path.join(llama_cpp, "llama-quantize")
        if not os.path.isfile(quantize_bin):
            quantize_bin = os.path.join(llama_cpp, "quantize")
        
        if os.path.isfile(quantize_bin):
            quantized_path = str(output_path).replace(".gguf", f"-{quantize}.gguf")
            cmd = [quantize_bin, str(output_path), quantized_path, quantize]
            print(f"[QUANTIZE] Running: {' '.join(cmd)}")
            subprocess.run(cmd)
            print(f"[QUANTIZE] Quantized model saved to {quantized_path}")
        else:
            print(f"[WARNING] Quantize binary not found. Skipping quantization.")
    
    return True


def export_gguf(model, output_path, llama_cpp_path=None, quantize=None):
    """Full export pipeline: FusionModel -> HF format -> GGUF.
    
    Args:
        model: FusionModel instance
        output_path: Output GGUF file path
        llama_cpp_path: Path to llama.cpp installation (optional)
        quantize: Quantization type (e.g. Q4_K_M)
    """
    output_path = Path(output_path)
    hf_dir = output_path.parent / "hf_intermediate"
    
    # Step 1: Export to HF format
    export_to_hf_format(model, hf_dir)
    
    # Step 2: Convert to GGUF
    success = convert_to_gguf(hf_dir, output_path, llama_cpp_path, quantize)
    
    if success:
        print(f"\n[DONE] GGUF export complete: {output_path}")
    else:
        print(f"\n[PARTIAL] HF format saved at {hf_dir}")
        print("  Install llama.cpp to complete GGUF conversion.")
    
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FusionModel to GGUF format")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output", default="output/gguf/model.gguf", help="Output GGUF path")
    parser.add_argument("--llama-cpp", default=None, help="Path to llama.cpp installation")
    parser.add_argument("--quantize", default=None, help="Quantization type (Q4_K_M, Q5_K_M, Q8_0)")
    args = parser.parse_args()
    
    from models.fusion_model import FusionModel
    model = FusionModel.from_pretrained(args.checkpoint)
    export_gguf(model, args.output, args.llama_cpp, args.quantize)
