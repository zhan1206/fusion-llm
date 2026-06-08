"""Minimal verification: does FusionModel.generate() work correctly with short sequences?"""
import sys, torch
sys.path.insert(0, '.')
from models.fusion_model import FusionConfig, FusionModel

torch.manual_seed(42)

config = FusionConfig(vocab_size=100, hidden_size=64, num_hidden_layers=2,
    num_attention_heads=4, intermediate_size=128, max_position_embeddings=32,
    block_size=8, latent_dim=8)
model = FusionModel(config)

# Train: input [2, x, y, 0, 0] -> labels [-100, x, y, 99, r]
# Very small task: x+y where x,y in [1,5], r in [2,10]
data = []
for x in range(1, 6):
    for y in range(1, 6):
        r = x + y
        data.append(([2, x, y, 0, 0], [-100, x, y, 99, r]))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for epoch in range(300):
    total_loss = 0
    for inp, lab in data:
        ids = torch.tensor([inp], dtype=torch.long)
        labs = torch.tensor([lab], dtype=torch.long)
        out = model(ids, labels=labs)
        total_loss += out.loss.item()
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss={total_loss/len(data):.4f}")

print(f"Final: loss={total_loss/len(data):.4f}")

# Test generate
model.eval()
correct = 0
total = 0
for x in range(1, 6):
    for y in range(1, 6):
        r = x + y
        inp = torch.tensor([[2, x, y]])
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=6, do_sample=False, pad_token_id=0)
        gen = out[0, 3:].tolist()
        # Check if gen starts with [99, result]
        predicted = gen[1] if len(gen) >= 2 and gen[0] == 99 else None
        ok = predicted == r
        correct += ok
        total += 1
        if x <= 2 and y <= 2:
            print(f"  {x}+{y}={r} | gen={gen} | pred={predicted} | {'OK' if ok else 'FAIL'}")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
