"""
Fusion Ollama deployment tool (v2 - fixed)

Features:
1. Auto-detect llama.cpp path
2. Convert HF model to GGUF format
3. Generate Modelfile
4. Create Ollama model
5. Support Thinking Dial control
6. Windows-compatible (shell=True for subprocess)

Usage:
    python inference/ollama_deploy_v2.py --model_path ./output/fusion-8b --model_name fusion-8b

Requirements:
    - llama.cpp (auto-detected or set LLAMA_CPP_DIR)
    - Ollama (https://ollama.com)

Author: 朱子瞻 (Zhu Zizhan)
Project: Fusion - Hexagonal Open-source LLM
License: Apache 2.0
"""

import argparse
import subprocess
import os
import json
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_llama_cpp() -> str:
    """
    Auto-detect llama.cpp directory
    
    Returns:
        llama.cpp directory path
        
    Raises:
        FileNotFoundError: llama.cpp not found
    """
    # 1. Check environment variable
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")
    if llama_cpp_dir and os.path.exists(llama_cpp_dir):
        convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        if os.path.exists(convert_script):
            logger.info(f"Found llama.cpp from env: {llama_cpp_dir}")
            return llama_cpp_dir
    
    # 2. Check common paths
    possible_paths = [
        "./llama.cpp",
        os.path.expanduser("~/llama.cpp"),
        "C:/llama.cpp",
        "D:/llama.cpp",
        os.path.join(os.path.dirname(__file__), "..", "llama.cpp"),
        os.path.join(os.path.dirname(__file__), "..", "..", "llama.cpp"),
    ]
    
    for path in possible_paths:
        path = os.path.abspath(path)
        convert_script = os.path.join(path, "convert-hf-to-gguf.py")
        if os.path.exists(convert_script):
            logger.info(f"Auto-detected llama.cpp: {path}")
            return path
    
    # 3. Not found
    raise FileNotFoundError(
        "llama.cpp not found. Set LLAMA_CPP_DIR or download to common path.\n"
        "Download: https://github.com/ggerganov/llama.cpp"
    )


def check_dependencies() -> bool:
    """
    Check dependencies (auto-detect llama.cpp + Windows compatible)
    
    Returns:
        Whether dependencies are satisfied
    """
    logger.info("Checking dependencies...")
    
    # 1. Check llama.cpp
    try:
        llama_cpp_dir = find_llama_cpp()
        convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        logger.info(f"llama.cpp convert script found: {convert_script}")
    except FileNotFoundError as e:
        logger.error(f"llama.cpp not found: {e}")
        return False
    
    # 2. Check Ollama (Windows needs shell=True)
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            shell=True,  # Windows needs shell=True
            timeout=10,
        )
        if result.returncode == 0:
            logger.info(f"Ollama installed: {result.stdout.strip()}")
        else:
            logger.warning("Ollama not installed or not working")
            logger.warning("Please install from https://ollama.com")
            return False
    except FileNotFoundError:
        logger.warning("Ollama not installed")
        logger.warning("Please install from https://ollama.com")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Ollama check timeout")
        return False
    
    logger.info("All dependencies satisfied")
    return True


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantize: str = "q4_k_m",
) -> str:
    """
    Convert HuggingFace model to GGUF format
    
    Args:
        model_path: HuggingFace model path
        output_path: Output path
        quantize: Quantization level (q4_k_m, q5_k_m, q8_0, etc.)
        
    Returns:
        Converted GGUF file path
    """
    logger.info("Converting to GGUF format...")
    
    llama_cpp_dir = find_llama_cpp()
    convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Conversion command
    cmd = [
        sys.executable,  # Use current Python interpreter
        convert_script,
        model_path,
        "--outtype", "f16",  # Convert to f16 first
        "--outfile", output_path,
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,  # Windows needs shell=True
        timeout=600,  # 10 minutes timeout
    )
    
    if result.returncode != 0:
        logger.error(f"Conversion failed: {result.stderr}")
        raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
    
    logger.info(f"GGUF conversion complete: {output_path}")
    
    # Quantization (optional)
    if quantize and quantize != "f16":
        logger.info(f"Quantizing model ({quantize})...")
        
        quantized_path = output_path.replace(".gguf", f"_{quantize}.gguf")
        quantize_cmd = [
            os.path.join(llama_cpp_dir, "llama-quantize"),
            output_path,
            quantized_path,
            quantize,
        ]
        
        result = subprocess.run(
            quantize_cmd,
            capture_output=True,
            text=True,
            shell=True,
            timeout=300,
        )
        
        if result.returncode != 0:
            logger.warning(f"Quantization failed: {result.stderr}")
            logger.warning("Using unquantized model")
        else:
            output_path = quantized_path
            logger.info(f"Quantization complete: {output_path}")
    
    return output_path


def create_modelfile(
    model_path: str,
    modelfile_path: str,
    model_name: str,
    context_size: int = 32768,
    thinking_dial: bool = True,
):
    """
    Create Ollama Modelfile
    
    Args:
        model_path: GGUF model path
        modelfile_path: Modelfile output path
        model_name: Model name
        context_size: Context window size
        thinking_dial: Whether to enable Thinking Dial
    """
    logger.info("Creating Modelfile...")
    
    # Get absolute path of model file
    model_path_abs = os.path.abspath(model_path)
    
    # Modelfile content
    content = f"""# Fusion Model: {model_name}
# Auto-generated by Fusion project
# Project: https://github.com/zhan1206/fusion-llm

FROM {model_path_abs}

# Model parameters
PARAMETER num_ctx {context_size}
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

# System prompt
SYSTEM \"\"\"You are a powerful AI assistant. You support dynamic reasoning intensity control:

- Simple questions: direct answer
- Complex questions: enable chain-of-thought reasoning

Use <|think| depth=N|> to control reasoning depth (N=0-3).
\"\"\"

# Template (supports Thinking Dial)
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
"""
    
    # If Thinking Dial is enabled, add special token handling
    if thinking_dial:
        content += f"""
# Thinking Dial examples (injected during training)
# <|think| depth=0|> Simple question, direct answer
# <|think| depth=3|> Complex question, detailed reasoning
"""
    
    # Write file
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Modelfile created: {modelfile_path}")


def create_ollama_model(modelfile_path: str, model_name: str) -> bool:
    """
    Create Ollama model using Modelfile
    
    Args:
        modelfile_path: Modelfile path
        model_name: Model name
        
    Returns:
        Whether creation succeeded
    """
    logger.info(f"Creating Ollama model: {model_name}...")
    
    # Remove existing model
    subprocess.run(
        ["ollama", "rm", model_name],
        capture_output=True,
        shell=True,
    )
    
    # Create model
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=True,
        timeout=300,
    )
    
    if result.returncode != 0:
        logger.error(f"Creation failed: {result.stderr}")
        return False
    
    logger.info(f"Ollama model created: {model_name}")
    logger.info(f"Run `ollama run {model_name}` to start using")
    return True


def deploy(
    model_path: str,
    model_name: str,
    output_dir: str = "./ollama_output",
    quantize: str = "q4_k_m",
    context_size: int = 32768,
    thinking_dial: bool = True,
) -> bool:
    """
    Complete deployment pipeline
    
    Args:
        model_path: HuggingFace model path
        model_name: Model name
        output_dir: Output directory
        quantize: Quantization level
        context_size: Context window
        thinking_dial: Whether to enable Thinking Dial
        
    Returns:
        Whether deployment succeeded
    """
    logger.info("Starting Ollama deployment...")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model name: {model_name}")
    
    # 1. Check dependencies
    if not check_dependencies():
        logger.error("Dependency check failed")
        return False
    
    # 2. Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Convert to GGUF
    gguf_path = os.path.join(output_dir, f"{model_name}.gguf")
    try:
        gguf_path = convert_to_gguf(
            model_path=model_path,
            output_path=gguf_path,
            quantize=quantize,
        )
    except RuntimeError as e:
        logger.error(f"GGUF conversion failed: {e}")
        return False
    
    # 4. Create Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    create_modelfile(
        model_path=gguf_path,
        modelfile_path=modelfile_path,
        model_name=model_name,
        context_size=context_size,
        thinking_dial=thinking_dial,
    )
    
    # 5. Create Ollama model
    if not create_ollama_model(
        modelfile_path=modelfile_path,
        model_name=model_name,
    ):
        logger.error("Ollama model creation failed")
        return False
    
    # 6. Generate usage example
    example_path = os.path.join(output_dir, "USAGE.md")
    generate_usage_example(model_name, example_path)
    
    logger.info("Deployment complete!")
    logger.info(f"Run: `ollama run {model_name}`")
    logger.info(f"Examples: see {example_path}")
    
    return True


def generate_usage_example(model_name: str, output_path: str):
    """
    Generate usage example document
    """
    content = f"""# Fusion Model Usage Examples

## 1. Basic Usage

```bash
# Start model
ollama run {model_name}

# Input question in interactive interface
> Explain quantum entanglement
```

## 2. Thinking Dial Control

Fusion supports dynamic reasoning intensity control. Add control token before question:

```bash
# depth=0: direct answer (casual chat, translation)
> <|think| depth=0|> How's the weather today?

# depth=1: simple reasoning
> <|think| depth=1|> Calculate 123 * 456

# depth=2: medium reasoning
> <|think| depth=2|> Prove Pythagorean theorem

# depth=3: deep reasoning (chain-of-thought)
> <|think| depth=3|> Solve this algorithm problem: ...
```

## 3. REST API

Ollama provides OpenAI-compatible API:

```bash
# Start Ollama service
ollama serve

# Call API
curl <a href="http://localhost:11434/api/generate">http://localhost:11434/api/generate</a> -d {{{{
  "model": "{model_name}",
  "prompt": "Explain machine learning",
  "stream": false
}}}}
```

## 4. Python Call

```python
import ollama

# Basic call
response = ollama.generate(
    model="{model_name}",
    prompt="Explain quantum entanglement",
)

print(response["response"])

# With Thinking Dial
response = ollama.generate(
    model="{model_name}",
    prompt="<|think| depth=2|> Prove Pythagorean theorem",
)

print(response["response"])
```

## 5. Parameter Tuning

Adjust generation parameters in Ollama:

```bash
# Temperature (creativity)
ollama run {model_name} --temperature 0.9

# Context window
ollama run {model_name} --num_ctx 16384

# Top-p sampling
ollama run {model_name} --top_p 0.95
```

---

**Tip**: See Ollama docs for more: https://ollama.com/docs
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Usage example generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fusion Ollama One-Click Deployment (v2)")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Ollama model name (e.g., fusion-8b)")
    parser.add_argument("--output_dir", type=str, default="./ollama_output",
                        help="Output directory")
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization level")
    parser.add_argument("--context_size", type=int, default=32768,
                        help="Context window size")
    parser.add_argument("--no_thinking_dial", action="store_false",
                        dest="thinking_dial",
                        help="Disable Thinking Dial")
    
    args = parser.parse_args()
    
    # Execute deployment
    success = deploy(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        quantize=args.quantize,
        context_size=args.context_size,
        thinking_dial=args.thinking_dial,
    )
    
    if success:
        logger.info("Deployment successful!")
    else:
        logger.error("Deployment failed")


if __name__ == "__main__":
    main()