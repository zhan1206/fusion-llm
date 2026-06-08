"""Quick test: check what model actually generates after training.
Uses short sequences (no padding) to avoid position index bugs.
"""
import sys, random, torch
sys.path.insert(0, '.')
from models.fusion_model import FusionConfig, FusionModel
from models.thinking_dial import ThinkingDialModel, ThinkingConfig

random.seed(42); torch.manual_seed(42)

config = FusionConfig(vocab_size=1000, hidden_size=256, num_hidden_layers=6,
                      num_attention_heads=8, intermediate_size=512,
                      max_position_embeddings=128, block_size=32, latent_dim=16)
model = FusionModel(config)
tc = ThinkingConfig(num_thinking_depths=4)
td = ThinkingDialModel(model, tc)
td.train()

data = [(x, 0, y, x+y) for x in range(1,51) for y in range(1,51)]
optimizer = torch.optim.AdamW(td.parameters(), lr=1e-3)

for epoch in range(200):
    batch = random.sample(data, 16)
    inps, tgts = [], []
    for x, op, y, r in batch:
        # Short sequence: [2, x, op, y, 0, 0, 0] -> labels [-100, x, op, y, 99, r, 1]
        inps.append([2, x, op, y, 0, 0, 0])
        tgts.append([-100, x, op, y, 99, r, 1])
    ids = torch.tensor(inps, dtype=torch.long)
    labs = torch.tensor(tgts, dtype=torch.long)
    optimizer.zero_grad()
    o = td(ids, labels=labs)
    o.loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: loss={o.loss.item():.4f}")

td.eval()
test_cases = [(5, 0, 3, 8), (10, 0, 15, 25), (50, 0, 49, 99), (7, 0, 13, 20)]
for x, op, y, r in test_cases:
    inp = torch.tensor([[2, x, op, y]])  # Short input, no padding
    for depth in range(4):
        out = td.generate(inp, max_new_tokens=8, thinking_depth=depth, do_sample=False, pad_token_id=0)
        gen = out[0, 4:].tolist()[:8]  # Generated tokens after input
        result_tok = None
        if len(gen) >= 2 and gen[0] == 99:
            result_tok = gen[1]
        match = "YES" if result_tok == r else "NO"
        print(f"  {x}+{y}={r} | depth={depth} | gen={gen} | result_tok={result_tok} | {match}")
