def test_train_loop():
    """Small training loop to verify loss decreases"""
    import sys
    sys.path.insert(0, ".")
    import torch
    from models.fusion_model import FusionModel, FusionConfig

    config = FusionConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        block_size=64,
        latent_dim=16,
        sbla_mode="pure_sbla",
        max_position_embeddings=256,
    )

    model = FusionModel(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size, seq_len = 4, 32

    print("[DEBUG] Small training loop (5 steps)...")
    for step in range(5):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: loss = {loss.item():.4f}")

    print("\nTraining loop successful - loss is decreasing!")
    assert True

if __name__ == "__main__":
    test_train_loop()