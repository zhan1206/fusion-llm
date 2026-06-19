"""
Fusion-LLM Hugging Face Spaces Demo
=====================================

Entry point for HF Spaces deployment. Provides:
1. Training tab - configure and train a FusionMini model on your data
2. Chat tab - generate text with the trained model

Run locally: python spaces/app.py
Deploy to HF Spaces: push this repo, set spaces/app.py as the app file.
"""

import gradio as gr
import torch
import sys
from pathlib import Path
import json
import random
import numpy as np
import time

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.fusion_mini import FusionMini, FusionMiniConfig

# ---- Global state ----
_model = None
_config = None
_tokenizer_map = None  # char_to_idx fallback
_vocab_size = 500
_device = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
#  DEMO DATA  (Chinese AI Q&A pairs for training)
# ================================================================

DEMO_DATA = [
    {"prompt": "什么是人工智能？", "response": "人工智能是让计算机模仿人类智能的技术。"},
    {"prompt": "机器学习是什么？", "response": "机器学习是AI的一个分支，让计算机从数据中学习模式。"},
    {"prompt": "深度学习怎么工作？", "response": "深度学习使用多层神经网络从大量数据中自动提取特征。"},
    {"prompt": "什么是自然语言处理？", "response": "自然语言处理是让计算机理解人类语言的技术。"},
    {"prompt": "Transformer架构的核心是什么？", "response": "Transformer的核心是自注意力机制，可以并行处理序列数据。"},
    {"prompt": "什么是GPU？", "response": "GPU是图形处理器，擅长并行计算，广泛用于深度学习训练。"},
    {"prompt": "Python是什么语言？", "response": "Python是一种简洁易读的高级编程语言，在AI领域使用广泛。"},
    {"prompt": "什么是神经网络？", "response": "神经网络是受人脑启发的计算模型，由多层神经元连接组成。"},
    {"prompt": "数据科学做什么？", "response": "数据科学从数据中提取洞见，结合统计学、编程和领域知识。"},
    {"prompt": "什么是强化学习？", "response": "强化学习是智能体通过与环境交互、试错来学习最优策略的方法。"},
    {"prompt": "什么是大语言模型？", "response": "大语言模型是参数规模巨大的神经网络，能够理解和生成人类语言。"},
    {"prompt": "AI会产生意识吗？", "response": "目前AI只是模式匹配工具，没有真正的意识。这是开放的科学问题。"},
    {"prompt": "什么是计算机视觉？", "response": "计算机视觉是让计算机理解图像和视频的技术，如人脸识别、目标检测。"},
    {"prompt": "什么是注意力机制？", "response": "注意力机制让模型关注输入中的重要部分，是Transformer成功的关键。"},
    {"prompt": "什么是过拟合？", "response": "过拟合是模型在训练数据上表现好但在新数据上表现差的现象。"},
    {"prompt": "什么是反向传播？", "response": "反向传播是计算神经网络梯度的算法，是深度学习训练的基础。"},
    {"prompt": "什么是迁移学习？", "response": "迁移学习将一个任务上学到的知识应用到另一个相关任务上。"},
    {"prompt": "什么是卷积神经网络？", "response": "卷积神经网络擅长处理图像，使用卷积核提取空间特征。"},
    {"prompt": "什么是生成对抗网络？", "response": "生成对抗网络由生成器和判别器组成，两者对抗训练生成逼真数据。"},
    {"prompt": "什么是元学习？", "response": "元学习是让模型学会如何学习，能快速适应新任务。"},
    {"prompt": "什么是自监督学习？", "response": "自监督学习从无标签数据中创建伪标签来训练模型。"},
    {"prompt": "AI会取代人类吗？", "response": "AI会改变很多工作方式，但创造力、情感理解等人类特质难以被替代。"},
    {"prompt": "什么是Tokenization？", "response": "Tokenization是将文本切分为模型可处理的词元的过程。"},
    {"prompt": "什么是Embedding？", "response": "Embedding是将词语或token映射为稠密向量的技术，捕获语义信息。"},
    {"prompt": "什么是损失函数？", "response": "损失函数衡量模型预测与真实值的差距，指导模型优化方向。"},
    {"prompt": "为什么需要激活函数？", "response": "激活函数引入非线性，让神经网络能学习复杂模式。"},
    {"prompt": "什么是批归一化？", "response": "批归一化对每层输入做标准化，加速训练并提升模型稳定性。"},
    {"prompt": "什么是Dropout？", "response": "Dropout在训练中随机丢弃神经元，防止模型过拟合。"},
    {"prompt": "什么是分布式训练？", "response": "分布式训练将模型或数据分布到多台机器/GPU上并行训练。"},
    {"prompt": "什么是模型压缩？", "response": "模型压缩通过剪枝、量化、蒸馏等手段减小模型体积和推理时间。"},
]


# ================================================================
#  TOKENIZER  (character-level fallback for simplicity)
# ================================================================

def build_tokenizer(texts):
    """Build a character-level index from a list of texts."""
    all_chars = set()
    for t in texts:
        all_chars.update(list(t))
    char_to_idx = {c: i + 4 for i, c in enumerate(sorted(all_chars))}
    char_to_idx["<PAD>"] = 0
    char_to_idx["<BOS>"] = 1
    char_to_idx["<EOS>"] = 2
    char_to_idx["<UNK>"] = 3
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode(text, char_to_idx, max_length=128):
    ids = [char_to_idx.get(c, 3) for c in text[:max_length]]
    ids = ids + [0] * (max_length - len(ids))
    return torch.tensor([ids], dtype=torch.long)


def decode(ids, idx_to_char):
    chars = [idx_to_char.get(i, "?") for i in ids if i > 0]
    return "".join(chars)


# ================================================================
#  MODEL HELPERS
# ================================================================

def create_model(vocab_size=500, hidden_size=256, num_layers=6, num_heads=8):
    config = FusionMiniConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        max_position_embeddings=128,
    )
    model = FusionMini(config)
    n_params = sum(p.numel() for p in model.parameters())
    return model, config, n_params


def train_model(
    model, config, data, char_to_idx, idx_to_char,
    num_epochs=5, batch_size=4, lr=5e-4, device="cuda",
    progress=gr.Progress()
):
    """Train the FusionMini model and return loss history."""
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim

    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    class SimpleDataset(Dataset):
        def __init__(self, items, char_to_idx, max_len=128):
            self.items = items
            self.char_to_idx = char_to_idx
            self.max_len = max_len

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            text = self.items[i]["prompt"] + " " + self.items[i]["response"]
            ids = [self.char_to_idx.get(c, 3) for c in text[:self.max_len]]
            ids = ids + [0] * (self.max_len - len(ids))
            t = torch.tensor(ids, dtype=torch.long)
            return {"input_ids": t, "attention_mask": (t != 0).long(), "labels": t.clone()}

    dataset = SimpleDataset(data, char_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(loader))

    model = model.to(device)
    model.train()
    loss_history = []

    total_steps = num_epochs * len(loader)
    step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1
            loss_history.append(loss.item())
            progress(step / total_steps, desc=f"Epoch {epoch+1}/{num_epochs}  Loss: {loss.item():.4f}")

        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} complete, avg_loss={avg:.4f}")

    model.eval()
    return model, loss_history


# ================================================================
#  GENERATION
# ================================================================

def generate_text(model, prompt, char_to_idx, idx_to_char, max_len=128, max_new=80, temperature=0.8, device="cuda"):
    """Simple autoregressive generation."""
    model.eval()
    input_ids = encode(prompt, char_to_idx, max_len).to(device)
    generated = input_ids.tolist()[0]
    generated = [t for t in generated if t != 0]  # strip padding

    with torch.no_grad():
        for _ in range(max_new):
            cur = torch.tensor([generated[-max_len:]], dtype=torch.long).to(device)
            outputs = model(input_ids=cur, return_dict=True)
            logits = outputs["logits"][0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            # Top-p sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            cutoff = (cumsum > 0.95).nonzero(as_tuple=False)
            if len(cutoff) > 0:
                top_p_idx = cutoff[0].item() + 1
                sorted_probs = sorted_probs[:top_p_idx]
                sorted_indices = sorted_indices[:top_p_idx]
                sorted_probs = sorted_probs / sorted_probs.sum()

            next_id = sorted_indices[torch.multinomial(sorted_probs, 1)[0]].item()
            generated.append(next_id)

            if len(generated) >= max_len + max_new:
                break

    return decode(generated, idx_to_char)


# ================================================================
#  GRADIO UI CALLBACKS
# ================================================================

def on_train(
    vocab_size, hidden_size, num_layers, num_heads,
    num_epochs, batch_size, learning_rate,
    progress=gr.Progress()
):
    """Trains model and returns status + loss chart data."""
    global _model, _config, _tokenizer_map, _vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _vocab_size = vocab_size

    # Build tokenizer from demo data
    all_texts = [d["prompt"] + " " + d["response"] for d in DEMO_DATA]
    c2i, i2c = build_tokenizer(all_texts)
    _tokenizer_map = (c2i, i2c)

    actual_vocab = len(c2i) + 4  # pad for safety
    actual_vocab = max(actual_vocab, vocab_size)

    model, config, n_params = create_model(actual_vocab, hidden_size, num_layers, num_heads)
    _model, _config = model, config

    progress(0.0, desc="Starting training...")

    model, history = train_model(
        model, config, DEMO_DATA, c2i, i2c,
        num_epochs=num_epochs, batch_size=batch_size, lr=learning_rate,
        device=device, progress=progress
    )
    _model = model

    # Build loss plot data
    steps = list(range(1, len(history) + 1))
    loss_df = {"Step": steps, "Loss": history}

    msg = (
        f"Training complete!\n\n"
        f"Model: {n_params/1e6:.2f}M params | "
        f"Vocab: {actual_vocab} | "
        f"Hidden: {hidden_size} | "
        f"Layers: {num_layers} | "
        f"Heads: {num_heads}\n"
        f"Final loss: {history[-1]:.4f}\n"
        f"Device: {device.upper()}"
    )
    return msg, gr.LinePlot(value=loss_df, x="Step", y="Loss", title="Training Loss", width=600, height=350)


def on_chat(prompt, temperature):
    """Generate a response from the trained model."""
    global _model, _tokenizer_map

    if _model is None:
        return "Model not trained yet. Please train first in the Training tab."

    c2i, i2c = _tokenizer_map
    response = generate_text(_model, prompt, c2i, i2c, temperature=temperature, device=_device)
    # Try to extract the response portion (after the prompt)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response


def on_load_prompt(preset):
    """Load a preset prompt."""
    presets = {
        "什么是": "什么是深度学习？",
        "AI": "AI会产生意识吗？",
        "Python": "Python是什么语言？",
        "模型": "什么是大语言模型？",
    }
    return presets.get(preset, "什么是人工智能？")


# ================================================================
#  GRADIO UI
# ================================================================

CSS = """
.gradio-container { max-width: 950px !important; }
.tab-header { font-size: 1.2em; font-weight: bold; }
.train-btn { background: linear-gradient(135deg, #667eea, #764ba2) !important; border: none !important; color: white !important; }
.generate-btn { background: linear-gradient(135deg, #f093fb, #f5576c) !important; border: none !important; color: white !important; }
"""

with gr.Blocks(title="Fusion-LLM Demo", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Fusion-LLM: Training & Generation Demo
    **Custom Transformer architecture** with SBLA attention and Thinking Dial.
    Train a FusionMini model on Chinese AI Q&A, then chat with it!
    """)

    with gr.Tabs():
        # ---- TRAINING TAB ----
        with gr.TabItem("Train Model", id="train"):
            gr.Markdown("### Model Configuration")
            with gr.Row():
                with gr.Column(scale=1):
                    vocab_size = gr.Slider(100, 2000, value=500, step=50, label="Vocab Size")
                    hidden_size = gr.Slider(64, 512, value=256, step=64, label="Hidden Size")
                    num_layers = gr.Slider(2, 16, value=6, step=1, label="Num Layers")
                with gr.Column(scale=1):
                    num_heads = gr.Slider(2, 16, value=8, step=2, label="Num Heads")
                    num_epochs = gr.Slider(1, 20, value=5, step=1, label="Epochs")
                    batch_size = gr.Slider(2, 32, value=8, step=2, label="Batch Size")
                with gr.Column(scale=1):
                    learning_rate = gr.Number(value=5e-4, label="Learning Rate", precision=6)
                    gr.Markdown("")
                    train_btn = gr.Button("Start Training", variant="primary", elem_classes=["train-btn"])

            with gr.Row():
                train_status = gr.Textbox(label="Status", lines=6, interactive=False)
                loss_plot = gr.LinePlot(show_label=False)

            train_btn.click(
                fn=on_train,
                inputs=[vocab_size, hidden_size, num_layers, num_heads, num_epochs, batch_size, learning_rate],
                outputs=[train_status, loss_plot],
            )

        # ---- CHAT TAB ----
        with gr.TabItem("Chat", id="chat"):
            gr.Markdown("### Generate text after training")
            with gr.Row():
                preset_dd = gr.Dropdown(
                    ["什么是", "AI", "Python", "模型"],
                    label="Preset Prompts",
                    value="什么是",
                )
                temperature_chat = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
            with gr.Row():
                prompt_box = gr.Textbox(
                    label="Your Prompt",
                    value="什么是人工智能？",
                    lines=3,
                    placeholder="Type your prompt here...",
                )
            generate_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])
            response_box = gr.Textbox(label="Model Response", lines=8, interactive=False)

            preset_dd.change(fn=on_load_prompt, inputs=[preset_dd], outputs=[prompt_box])
            generate_btn.click(fn=on_chat, inputs=[prompt_box, temperature_chat], outputs=[response_box])

        # ---- INFO TAB ----
        with gr.TabItem("About", id="about"):
            gr.Markdown("""
            ## Fusion-LLM

            **A compact, interpretable Chinese LLM framework.**

            ### Architecture
            - **SBLA**: Sliding Block Latent Attention for efficient long-sequence processing
            - **Thinking Dial**: Dynamic inference depth control via special tokens
            - **Modular**: Swap components (attention, quantization, training strategy)

            ### This Demo
            - Trains FusionMini (a minimized proof-of-concept) on Chinese AI Q&A pairs
            - Character-level tokenizer for simplicity — production would use SentencePiece
            - Runs on free HF Spaces T4 (16GB VRAM)

            ### Links
            - [GitHub](https://github.com/zhan1206/fusion-llm)
            - [Paper (coming soon)]()

            ### License
            Apache 2.0
            """)

if __name__ == "__main__":
    print(f"Device: {_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
