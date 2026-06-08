"""
ThinkingDial vs Vanilla comparison - minimal working benchmark.
Short sequences, small model, simple addition task.
"""
import sys, random, torch
sys.path.insert(0, '.')
from models.fusion_model import FusionConfig, FusionModel
from models.thinking_dial import ThinkingDialModel, ThinkingConfig

torch.manual_seed(42); random.seed(42)

config = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128, max_position_embeddings=32,
    block_size=8, latent_dim=8)

def make_data(n, max_val=10):
    data = []
    for x in range(1, max_val + 1):
        for y in range(1, max_val + 1):
            data.append(([2, x, y, 0, 0], [-100, x, y, 99, x + y]))
    random.shuffle(data)
    return data[:n]

def train_model(model_or_td, data, epochs=200, lr=1e-2):
    optimizer = torch.optim.AdamW(
        [p for p in model_or_td.parameters() if p.requires_grad], lr=lr)
    model_or_td.train()
    for epoch in range(epochs):
        total = 0
        for inp, lab in data:
            ids = torch.tensor([inp], dtype=torch.long)
            labs = torch.tensor([lab], dtype=torch.long)
            out = model_or_td(ids, labels=labs)
            total += out.loss.item()
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={total/len(data):.4f}")
    model_or_td.eval()
    return total / len(data)

def eval_accuracy(model_fn, data):
    correct = total = 0
    for inp, lab in data:
        x_in = torch.tensor([inp[:3]], dtype=torch.long)  # [2, x, y] (no padding)
        gen = model_fn(x_in)
        tokens = gen[0, 3:].tolist()
        pred = tokens[1] if len(tokens) >= 2 and tokens[0] == 99 else None
        correct += pred == lab[4]  # lab[4] = result
        total += 1
    return correct, total

# === Vanilla Transformer ===
print("=" * 60)
print("Experiment: ThinkingDial vs Vanilla (addition, 1..10)")
print("=" * 60)

train_data = make_data(80)  # 80 training examples
test_data = make_data(20, max_val=10)  # test on same domain

# Train vanilla
print("\n--- Vanilla Transformer ---")
vanilla = FusionModel(config)
train_model(vanilla, train_data, epochs=200)
vc, vt = eval_accuracy(
    lambda x: vanilla.generate(x, max_new_tokens=6, do_sample=False, pad_token_id=0),
    test_data)
print(f"  Accuracy: {vc}/{vt} = {vc/vt*100:.1f}%")

# Train ThinkingDial (depth=0 = vanilla behavior)
print("\n--- ThinkingDial (depth=0, no thinking bias) ---")
tc = ThinkingConfig(num_thinking_depths=4)
base0 = FusionModel(config)
td0 = ThinkingDialModel(base0, tc)
train_model(td0, train_data, epochs=200)
c0, t0 = eval_accuracy(
    lambda x: td0.generate(x, max_new_tokens=6, thinking_depth=0, do_sample=False, pad_token_id=0),
    test_data)
print(f"  Accuracy: {c0}/{t0} = {c0/t0*100:.1f}%")

# Train ThinkingDial (depth=3, maximum thinking)
print("\n--- ThinkingDial (depth=3, max thinking) ---")
base3 = FusionModel(config)
td3 = ThinkingDialModel(base3, tc)
train_model(td3, train_data, epochs=200)
c3, t3 = eval_accuracy(
    lambda x: td3.generate(x, max_new_tokens=6, thinking_depth=3, do_sample=False, pad_token_id=0),
    test_data)
print(f"  Accuracy: {c3}/{t3} = {c3/t3*100:.1f}%")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Vanilla:             {vc}/{vt} = {vc/vt*100:.1f}%")
print(f"  ThinkingDial d=0:    {c0}/{t0} = {c0/t0*100:.1f}%")
print(f"  ThinkingDial d=3:    {c3}/{t3} = {c3/t3*100:.1f}%")
