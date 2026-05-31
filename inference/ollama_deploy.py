"""
Fusion 推理部署：Ollama 一键部署工具

功能：
1. 自动转换模型为 GGUF 格式
2. 生成 Modelfile
3. 创建 Ollama 模型
4. 支持 Thinking Dial 控制

使用方法：
    python inference/ollama_deploy.py --model_path ./output/fusion-8b --model_name fusion-8b

前置要求：
    - 安装 llama.cpp（用于转换）
    - 安装 Ollama（https://ollama.com）

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import subprocess
import os
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """
    检查依赖项
    """
    logger.info("🔍 检查依赖项...")
    
    # 检查 llama.cpp 转换脚本
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")
    convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
    
    if not os.path.exists(convert_script):
        logger.warning(f"⚠️  未找到 llama.cpp 转换脚本：{convert_script}")
        logger.warning("   请设置环境变量 LLAMA_CPP_DIR 或手动下载 llama.cpp")
        logger.warning("   下载地址：<ADDRESS_REMOVED>
        return False
    
    # 检查 Ollama
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"✅ Ollama 已安装：{result.stdout.strip()}")
        else:
            logger.warning("⚠️  Ollama 未安装或无法运行")
            logger.warning("   请访问 https://ollama.com 安装")
            return False
    except FileNotFoundError:
        logger.warning("⚠️  Ollama 未安装")
        logger.warning("   请访问 https://ollama.com 安装")
        return False
    
    logger.info("✅ 依赖项检查通过")
    return True


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantize: str = "q4_k_m",
):
    """
    将 HuggingFace 模型转换为 GGUF 格式
    
    参数：
        model_path: HuggingFace 模型路径
        output_path: 输出路径
        quantize: 量化级别（q4_k_m, q5_k_m, q8_0 等）
    """
    logger.info("🔄 转换为 GGUF 格式...")
    
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")
    convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
    
    # 转换命令
    cmd = [
        "python", convert_script,
        model_path,
        "--outtype", "f16",  # 先转为 f16
        "--outfile", output_path,
    ]
    
    logger.info(f"   运行命令：{' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ 转换失败：{result.stderr}")
        raise RuntimeError("GGUF 转换失败")
    
    logger.info(f"✅ GGUF 转换完成：{output_path}")
    
    # 量化（可选）
    if quantize:
        logger.info(f"🔧 量化模型（{quantize}）...")
        
        quantize_cmd = [
            os.path.join(llama_cpp_dir, "llama-quantize"),
            output_path,
            output_path.replace(".gguf", f"_{quantize}.gguf"),
            quantize,
        ]
        
        result = subprocess.run(quantize_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"⚠️  量化失败：{result.stderr}")
            logger.warning("   继续使用未量化模型")
        else:
            output_path = output_path.replace(".gguf", f"_{quantize}.gguf")
            logger.info(f"✅ 量化完成：{output_path}")
    
    return output_path


def create_modelfile(
    model_path: str,
    modelfile_path: str,
    model_name: str,
    context_size: int = 32768,
    thinking_dial: bool = True,
):
    """
    创建 Ollama Modelfile
    
    参数：
        model_path: GGUF 模型路径
        modelfile_path: Modelfile 输出路径
        model_name: 模型名称
        context_size: 上下文窗口大小
        thinking_dial: 是否启用 Thinking Dial
    """
    logger.info("📝 创建 Modelfile...")
    
    # Modelfile 内容
    content = f"""# Fusion 模型：{model_name}
# 由 Fusion 项目自动生成
# 项目地址：<ADDRESS_REMOVED>

FROM {model_path}

# 模型参数
PARAMETER num_ctx {context_size}
PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

# 系统提示词
SYSTEM \"\"\"你是一个强大的 AI 助手。你支持动态推理强度控制：

- 简单问题：直接回答
- 复杂问题：启用思维链推理

使用 <|think| depth=N|> 控制推理深度（N=0-3）。
\"\"\"

# 模板（支持 Thinking Dial）
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
\"\"\"
"""
    
    # 如果启用 Thinking Dial，添加特殊 token 处理
    if thinking_dial:
        content += f"""
# Thinking Dial 示例（训练时注入）
# <|think_depth_0|> 简单问题，直接回答
# <|think_depth_3|> 复杂问题，详细推理
"""
    
    # 写入文件
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"✅ Modelfile 创建完成：{modelfile_path}")


def create_ollama_model(modelfile_path: str, model_name: str):
    """
    使用 Modelfile 创建 Ollama 模型
    
    参数：
        modelfile_path: Modelfile 路径
        model_name: 模型名称
    """
    logger.info(f"🚀 创建 Ollama 模型：{model_name}...")
    
    # 删除已存在的模型
    subprocess.run(
        ["ollama", "rm", model_name],
        capture_output=True,
    )
    
    # 创建模型
    cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    
    logger.info(f"   运行命令：{' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ 创建失败：{result.stderr}")
        raise RuntimeError("Ollama 模型创建失败")
    
    logger.info(f"✅ Ollama 模型创建成功：{model_name}")
    logger.info(f"   运行 `ollama run {model_name}` 开始使用")


def deploy(
    model_path: str,
    model_name: str,
    output_dir: str = "./ollama_output",
    quantize: str = "q4_k_m",
    context_size: int = 32768,
    thinking_dial: bool = True,
):
    """
    完整部署流程
    
    参数：
        model_path: HuggingFace 模型路径
        model_name: 模型名称
        output_dir: 输出目录
        quantize: 量化级别
        context_size: 上下文窗口
        thinking_dial: 是否启用 Thinking Dial
    """
    logger.info("🚀 开始 Ollama 部署流程...")
    logger.info(f"   模型路径：{model_path}")
    logger.info(f"   模型名称：{model_name}")
    
    # 1. 检查依赖
    if not check_dependencies():
        logger.error("❌ 依赖项检查失败，请先安装所需工具")
        return False
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 转换为 GGUF
    gguf_path = os.path.join(output_dir, f"{model_name}.gguf")
    try:
        gguf_path = convert_to_gguf(
            model_path=model_path,
            output_path=gguf_path,
            quantize=quantize,
        )
    except RuntimeError as e:
        logger.error(f"❌ GGUF 转换失败：{e}")
        return False
    
    # 4. 创建 Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    create_modelfile(
        model_path=gguf_path,
        modelfile_path=modelfile_path,
        model_name=model_name,
        context_size=context_size,
        thinking_dial=thinking_dial,
    )
    
    # 5. 创建 Ollama 模型
    try:
        create_ollama_model(
            modelfile_path=modelfile_path,
            model_name=model_name,
        )
    except RuntimeError as e:
        logger.error(f"❌ Ollama 模型创建失败：{e}")
        return False
    
    # 6. 生成使用示例
    example_path = os.path.join(output_dir, "USAGE.md")
    generate_usage_example(model_name, example_path)
    
    logger.info("✅ 部署完成！")
    logger.info(f"   运行：`ollama run {model_name}`")
    logger.info(f"   示例：见 {example_path}")
    
    return True


def generate_usage_example(model_name: str, output_path: str):
    """
    生成使用示例文档
    """
    content = f"""# Fusion 模型使用示例

## 1. 基础使用

```bash
# 启动模型
ollama run {model_name}

# 在交互界面中输入问题
> 解释量子纠缠
```

## 2. Thinking Dial 控制

Fusion 支持动态推理强度控制。在问题前添加控制 token：

```bash
# depth=0：直接回答（闲聊、翻译）
> <|think_depth_0|> 今天天气怎么样？

# depth=1：简单推理
> <|think_depth_1|> 计算 123 * 456

# depth=2：中等推理
> <|think_depth_2|> 证明勾股定理

# depth=3：深度推理（思维链）
> <|think_depth_3|> 解决这个算法问题：...
```

## 3. REST API

Ollama 提供 OpenAI 兼容的 API：

```bash
# 启动 Ollama 服务
ollama serve

# 调用 API
curl <a href="http://localhost:11434/api/generate">http://localhost:11434/api/generate</a> -d {{
  "model": "{model_name}",
  "prompt": "解释机器学习",
  "stream": false
}}
```

## 4. Python 调用

```python
import ollama

# 基础调用
response = ollama.generate(
    model="{model_name}",
    prompt="解释量子纠缠",
)

print(response["response"])

# 带 Thinking Dial
response = ollama.generate(
    model="{model_name}",
    prompt="<|think_depth_2|> 证明勾股定理",
)

print(response["response"])
```

## 5. 参数调整

在 Ollama 中调整生成参数：

```bash
# 温度（创造性）
ollama run {model_name} --temperature 0.9

# 上下文窗口
ollama run {model_name} --num_ctx 16384

# Top-p 采样
ollama run {model_name} --top_p 0.95
```

---

**提示**：更多用法见 Ollama 文档 https://ollama.com/docs
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"📖 使用示例已生成：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fusion Ollama 一键部署")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace 模型路径")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Ollama 模型名称（如 fusion-8b）")
    parser.add_argument("--output_dir", type=str, default="./ollama_output",
                        help="输出目录")
    parser.add_argument("--quantize", type=str, default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="量化级别")
    parser.add_argument("--context_size", type=int, default=32768,
                        help="上下文窗口大小）")
    parser.add_argument("--no_thinking_dial", action="store_false",
                        dest="thinking_dial",
                        help="禁用 Thinking Dial")
    
    args = parser.parse_args()
    
    # 执行部署
    success = deploy(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        quantize=args.quantize,
        context_size=args.context_size,
        thinking_dial=args.thinking_dial,
    )
    
    if success:
        logger.info("🎉 部署成功！")
    else:
        logger.error("❌ 部署失败")


if __name__ == "__main__":
    main()
