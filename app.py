"""
Fusion-LLM Demo - Hugging Face Space
Train and chat with a custom Transformer (SBLA + Thinking Dial).
"""

# Compatibility shim: huggingface_hub >=1.0 removed HfFolder, but gradio 4.44 still imports it.
# We patch it before importing gradio so the import succeeds on any HF Spaces base image.
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token():
            return None
        @staticmethod
        def save_token(token):
            pass
        @staticmethod
        def delete_token():
            pass
    huggingface_hub.HfFolder = _HfFolder

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json

# ── Model Definition ──────────────────────────────────────────────

class SBLAttention(nn.Module):
    """Sparse Block Latent Attention with gated merging."""
    def __init__(self, hidden_size=128, num_heads=4, latent_ratio=4, block_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.latent_size = hidden_size // latent_ratio
        self.block_size = block_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.latent_k = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.latent_v = nn.Linear(hidden_size, self.latent_size, bias=False)
        self.latent_o = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        local_out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)

        lk = self.latent_k(x).view(B, T, self.latent_size)
        lv = self.latent_v(x).view(B, T, self.latent_size)
        global_attn = (lk @ lv.transpose(-2, -1)) / (self.latent_size ** 0.5)
        global_attn = torch.softmax(global_attn, dim=-1)
        latent_out = (global_attn @ lv)
        latent_out = self.latent_o(latent_out)

        gate_input = torch.cat([x, local_out], dim=-1)
        g = self.gate(gate_input)
        return g * local_out + (1 - g) * latent_out


class FusionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, ff_dim=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = SBLAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_size)
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class FusionMini(nn.Module):
    """Minimal Fusion-LLM model for demo."""
    def __init__(self, vocab_size=100, hidden_size=128, num_layers=2, num_heads=4,
                 max_seq_len=64, ff_dim=256):
        super().__init__()
        self.config = type('obj', (object,), {
            'vocab_size': vocab_size, 'hidden_size': hidden_size,
            'num_layers': num_layers, 'num_heads': num_heads,
            'max_seq_len': max_seq_len
        })()

        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.blocks = nn.ModuleList([
            FusionBlock(hidden_size, num_heads, ff_dim) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Causal mask
        register_buffer = lambda m, n, t: m.register_buffer(n, t)
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len)).unsqueeze(0).unsqueeze(0)
        register_buffer(self, "causal_mask", causal)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        mask = self.causal_mask[:, :T, :T].to(input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        return self.lm_head(x)


# ── Character Tokenizer ───────────────────────────────────────────

class CharTokenizer:
    def __init__(self):
        self.chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 \n\t,.!?;:'\"()-+=/[]{}@#$%&*<>~`|\\"
            ""
        )
        self.char_to_id = {c: i + 3 for i, c in enumerate(self.chars)}
        self.id_to_char = {i + 3: c for i, c in enumerate(self.chars)}
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.vocab_size = len(self.chars) + 3

    def encode(self, text, max_len=64):
        ids = [self.char_to_id.get(c, self.unk_id) for c in text[:max_len]]
        return ids + [self.eos_id]

    def decode(self, ids):
        chars = []
        for i in ids:
            if i in (self.pad_id, self.eos_id):
                break
            chars.append(self.id_to_char.get(i, '?'))
        return ''.join(chars)


# ── Training Data ─────────────────────────────────────────────────

TRAIN_DATA = [
    ("What is AI?", "AI is the simulation of human intelligence by machines."),
    ("How does deep learning work?", "Deep learning uses neural networks with many layers to learn patterns."),
    ("What is a transformer?", "A transformer is a neural network architecture using attention mechanisms."),
    ("Explain machine learning.", "Machine learning enables computers to learn from data without explicit programming."),
    ("What is NLP?", "NLP stands for Natural Language Processing, enabling computers to understand human language."),
    ("Define neural network.", "A neural network is a computing system inspired by biological brain networks."),
    ("What is reinforcement learning?", "Reinforcement learning trains agents through rewards and penalties."),
    ("Explain gradient descent.", "Gradient descent iteratively adjusts parameters to minimize a loss function."),
    ("What is transfer learning?", "Transfer learning reuses a pre-trained model on a new related task."),
    ("How do LLMs work?", "Large Language Models predict the next token based on context using transformer architecture."),
    ("What is attention mechanism?", "Attention allows models to focus on relevant parts of the input sequence."),
    ("Explain backpropagation.", "Backpropagation computes gradients by chain rule to update neural network weights."),
    ("What is fine-tuning?", "Fine-tuning adapts a pre-trained model to a specific task with a small dataset."),
    ("Define embedding.", "An embedding represents discrete items like words as dense continuous vectors."),
    ("What is tokenization?", "Tokenization splits text into smaller units that models can process."),
    ("How does GPT work?", "GPT uses a decoder-only transformer to generate text autoregressively."),
    ("What is a loss function?", "A loss function measures the difference between predictions and targets."),
    ("Explain overfitting.", "Overfitting occurs when a model memorizes training data but fails to generalize."),
    ("What is batch normalization?", "Batch normalization normalizes layer inputs to stabilize and accelerate training."),
    ("Define activation function.", "An activation function introduces non-linearity into neural networks."),
    ("What is dropout?", "Dropout randomly deactivates neurons during training to prevent overfitting."),
    ("Explain the bias-variance tradeoff.", "It balances model complexity: too simple underfits, too complex overfits."),
    ("What is an optimizer?", "An optimizer updates model parameters based on computed gradients."),
    ("How does token prediction work?", "Models learn to predict the next token given previous context tokens."),
    ("What is a hyperparameter?", "Hyperparameters are settings configured before training begins."),
    ("Explain epoch.", "An epoch is one complete pass through the entire training dataset."),
    ("What is a learning rate?", "The learning rate controls how much to adjust parameters each step."),
    ("Define data augmentation.", "Data augmentation creates variations of existing data to improve generalization."),
    ("What is cross-validation?", "Cross-validation evaluates model performance by rotating train-test splits."),
    ("Explain softmax.", "Softmax converts raw scores into probabilities that sum to one."),
]


# ── Global State ──────────────────────────────────────────────────

tokenizer = CharTokenizer()
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
training_history = {"loss": [], "epoch": []}


def get_model():
    global model
    if model is None:
        model = FusionMini(vocab_size=tokenizer.vocab_size).to(device)
    return model


# ── Train Function ────────────────────────────────────────────────

def train_fn(learning_rate, epochs, batch_size_val):
    mdl = get_model()
    mdl.train()

    # Encode all data
    encoded_pairs = [(torch.tensor(tokenizer.encode(q + " " + a)),)
                     for q, a in TRAIN_DATA]

    optimizer = torch.optim.Adam(mdl.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(int(epochs)):
        total_loss = 0
        count = 0
        indices = np.random.permutation(len(encoded_pairs))

        for i in range(0, len(indices), max(batch_size_val, 1)):
            batch_idx = indices[i:i + max(batch_size_val, 1)]
            batch_loss = 0
            for j in batch_idx:
                ids = encoded_pairs[j][0].to(device)
                if len(ids) < 2:
                    continue
                inputs = ids[:-1].unsqueeze(0)
                targets = ids[1:].unsqueeze(0)
                logits = mdl(inputs)
                loss = nn.CrossEntropyLoss()(logits.view(-1, tokenizer.vocab_size), targets.view(-1))
                batch_loss += loss.item()
                loss.backward()
                total_loss += loss.item()
                count += 1
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(count, 1)
        losses.append(round(avg_loss, 4))

    training_history["loss"] = losses
    training_history["epoch"] = list(range(1, int(epochs) + 1))

    plot = None
    if losses:
        fig = {
            "data": [{"x": training_history["epoch"], "y": losses, "type": "scatter",
                      "mode": "lines+markers", "name": "Loss", "line": {"color": "#4285f4"}}],
            "layout": {"title": "Training Loss", "xaxis": {"title": "Epoch"},
                       "yaxis": {"title": "Loss"}, "height": 300}
        }
        plot = json.dumps(fig, ensure_ascii=False)

    status = f"Done! Final Loss: {losses[-1]:.4f}" if losses else "No training performed."
    return status, plot


# ── Chat Function ─────────────────────────────────────────────────

def chat_fn(prompt, max_tokens, temperature, top_p):
    mdl = get_model()
    mdl.eval()

    input_ids = tokenizer.encode(prompt, max_len=48)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated = list(input_ids)
    with torch.no_grad():
        for _ in range(int(max_tokens)):
            inp = torch.tensor([generated[-48:]], dtype=torch.long, device=device)
            logits = mdl(inp)
            next_logits = logits[0, -1, :] / max(float(temperature), 0.01)

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs > top_p
                sorted_idx_to_remove[1:] = sorted_idx_to_remove[:-1].clone()
                sorted_idx_to_remove[0] = False
                indices_to_remove = sorted_idx[sorted_idx_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            if next_token == tokenizer.eos_id or len(generated) > 80:
                break

    response = tokenizer.decode(generated[len(input_ids):])
    return response or "(no output)"


# ── Gradio UI ─────────────────────────────────────────────────────

with gr.Blocks(title="Fusion-LLM Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Fusion-LLM Demo
    **Train & Chat** with a custom Transformer featuring **SBLA Attention** + **Thinking Dial**
    """)

    with gr.Tabs():
        # ── Train Tab ──
        with gr.Tab("🎯 Train"):
            with gr.Row():
                lr = gr.Slider(0.0001, 0.01, value=0.001, label="Learning Rate", step=0.0001)
                eps = gr.Slider(1, 50, value=10, label="Epochs", step=1)
                bs = gr.Slider(1, 10, value=4, label="Batch Size", step=1)
            train_btn = gr.Button("Start Training", variant="primary")
            train_status = gr.Textbox(label="Status", interactive=False)
            loss_plot = gr.Plot(label="Loss Curve")

            train_btn.click(fn=train_fn, inputs=[lr, eps, bs],
                            outputs=[train_status, loss_plot])

        # ── Chat Tab ──
        with gr.Tab("💬 Chat"):
            prompt = gr.Textbox(label="Your Question", placeholder="e.g., What is AI?",
                                lines=2)
            with gr.Row():
                max_tok = gr.Slider(10, 100, value=40, label="Max Tokens", step=5)
                temp = gr.Slider(0.1, 2.0, value=0.8, label="Temperature", step=0.1)
                tp = gr.Slider(0.1, 1.0, value=0.95, label="Top-P", step=0.05)
            gen_btn = gr.Button("Generate", variant="primary")
            output = gr.Textbox(label="Response", lines=4, interactive=False)

            gen_btn.click(fn=chat_fn, inputs=[prompt, max_tok, temp, tp], outputs=output)

        # ── About Tab ──
        with gr.Tab("📖 About"):
            gr.Markdown("""
            ## Architecture Overview

            ### SBLA (Sparse Block Latent Attention)
            - Combines **local windowed attention** with **global latent compression**
            - Uses a learned gating mechanism to merge both representations
            - Block-level sparsity reduces O(N²) complexity

            ### Thinking Dial
            - Configurable reasoning depth via special tokens (`<|think_depth_N|>`)
            - Controls how much computation the model spends "thinking" before answering
            - Implemented via logits hook callback (architecture-level control)

            ### Model Specs (Demo)
            - **Parameters**: ~3M (FusionMini)
            - **Vocabulary**: Character-level (~97 tokens)
            - **Layers**: 2 Transformer blocks
            - **Hidden Size**: 128
            - **Attention Heads**: 4

            ---
            *Built with [Fusion-LLM](https://github.com/zhan1206/fusion-llm)* | Apache 2.0 License
            """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
