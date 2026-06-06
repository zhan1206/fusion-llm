def test_training():
    """Test training"""
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

    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    print(f"Loss: {outputs['loss'].item():.4f}")

    loss = outputs["loss"]
    loss.backward()
    print(f"Gradients exist: {model.embeddings.weight.grad is not None}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    outputs["loss"].backward()
    optimizer.step()
    print("Optimizer step successful!")
    assert True

if __name__ == "__main__":
    test_training()