#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Script for Fusion-LLM

Implements DPO alignment training using preference pairs (chosen vs rejected).
Based on Rafailov et al. (2023) "Direct Preference Optimization".

Usage:
    python train/dpo_finetune.py \
        --model_path output/mini_model \
        --data_path data/preference_data.json \
        --output_path output/dpo_model \
        --epochs 3 \
        --lr 1e-6 \
        --beta 0.1

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import FusionModel, FusionConfig


@dataclass
class DPOConfig:
    """DPO training configuration."""
    model_path: str = "output/mini_model"
    data_path: str = "data/preference_data.json"
    output_path: str = "output/dpo_model"
    beta: float = 0.1          # DPO temperature parameter
    lr: float = 1e-6           # Learning rate
    epochs: int = 3
    batch_size: int = 2
    max_seq_len: int = 256
    warmup_steps: int = 50
    gradient_accumulation: int = 4
    save_every: int = 100      # Save checkpoint every N steps
    config_overrides: str = "" # JSON config overrides


class DPOTrainer:
    """
    DPO Trainer for Fusion models.
    
    Minimizes the DPO loss:
        L_DPO = -E[log sigmoid(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]
    
    Where:
        - pi: current policy (model being trained)
        - pi_ref: reference policy (frozen copy of initial model)
        - y_w: chosen (preferred) response
        - y_l: rejected response
    """
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        
    def load_model(self) -> tuple:
        """Load model and create reference copy."""
        config_path = Path(self.config.model_path)
        
        if config_path.joinpath("config.json").exists():
            model_config = FusionConfig.from_pretrained(str(config_path))
        else:
            # Use mini config as default
            model_config = FusionConfig(
                vocab_size=10000, hidden_size=256, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=512,
                block_size=64, latent_dim=16, max_position_embeddings=256,
            )
        
        # Apply overrides
        if self.config.config_overrides:
            overrides = json.loads(self.config.config_overrides)
            for k, v in overrides.items():
                setattr(model_config, k, v)
        
        # Policy model (will be trained)
        policy = FusionModel(model_config)
        
        # Load weights if available
        weight_path = config_path / "final_model.pth"
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            policy.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weight_path}")
        
        policy = policy.to(self.device)
        
        # Reference model (frozen copy)
        ref_policy = FusionModel(model_config)
        ref_policy.load_state_dict(policy.state_dict())
        ref_policy = ref_policy.to(self.device)
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad = False
        
        return policy, ref_policy
    
    def load_data(self) -> list:
        """Load preference data."""
        data_path = Path(self.config.data_path)
        if not data_path.exists():
            print(f"No preference data at {data_path}, generating synthetic data...")
            return self._generate_synthetic_data()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} preference pairs from {data_path}")
        return data
    
    def _generate_synthetic_data(self) -> list:
        """Generate synthetic preference data for testing."""
        pairs = []
        templates = [
            {"prompt": "What is machine learning", "chosen": "Machine learning is a subset of AI that enables systems to learn from data.", "rejected": "ML is just statistics."},
            {"prompt": "Explain neural networks", "chosen": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.", "rejected": "They are like brains."},
            {"prompt": "What is deep learning", "chosen": "Deep learning uses multi-layered neural networks to automatically learn hierarchical representations from data.", "rejected": "It is deep ML."},
            {"prompt": "What is Python", "chosen": "Python is a high-level, interpreted programming language known for its readability and extensive ecosystem of libraries.", "rejected": "A snake."},
            {"prompt": "Explain gradient descent", "chosen": "Gradient descent is an optimization algorithm that iteratively moves parameters in the direction of steepest decrease of the loss function.", "rejected": "Going downhill."},
        ]
        
        for t in templates:
            pairs.append({
                "prompt": t["prompt"],
                "chosen": t["chosen"],
                "rejected": t["rejected"],
            })
        
        # Save for reuse
        out_path = Path("data/preference_data.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"Generated {len(pairs)} synthetic pairs, saved to {out_path}")
        
        return pairs
    
    def _tokenize(self, text: str, max_len: int) -> torch.Tensor:
        """Simple character-level tokenization."""
        encoded = list(text.encode('utf-8'))[:max_len]
        padded = encoded + [0] * (max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long)
    
    def _get_log_probs(self, model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of sequences under model."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        per_token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        
        # Mask padding
        mask = (shift_labels != 0).float()
        return (per_token_log_probs * mask).sum(dim=1)
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DPO loss."""
        chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        return loss
    
    def train(self):
        """Run DPO training."""
        policy, ref_policy = self.load_model()
        data = self.load_data()
        
        optimizer = torch.optim.AdamW(policy.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=self.config.warmup_steps
        )
        
        output_dir = Path(self.config.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[DPO] Starting training:")
        print(f"  Model: {self.config.model_path}")
        print(f"  Data: {len(data)} pairs")
        print(f"  Beta: {self.config.beta}")
        print(f"  LR: {self.config.lr}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Device: {self.device}")
        
        policy.train()
        global_step = 0
        
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Shuffle data each epoch
            indices = torch.randperm(len(data))
            
            for i in range(0, len(data), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                
                chosen_ids = []
                rejected_ids = []
                for idx in batch_indices:
                    item = data[idx.item()]
                    prompt = item['prompt']
                    chosen_ids.append(self._tokenize(prompt + " " + item['chosen'], self.config.max_seq_len))
                    rejected_ids.append(self._tokenize(prompt + " " + item['rejected'], self.config.max_seq_len))
                
                chosen_ids = torch.stack(chosen_ids).to(self.device)
                rejected_ids = torch.stack(rejected_ids).to(self.device)
                chosen_mask = (chosen_ids != 0).float()
                rejected_mask = (rejected_ids != 0).float()
                
                # Policy log probs
                policy_chosen_logps = self._get_log_probs(policy, chosen_ids, chosen_mask)
                policy_rejected_logps = self._get_log_probs(policy, rejected_ids, rejected_mask)
                
                # Reference log probs (no grad)
                with torch.no_grad():
                    ref_chosen_logps = self._get_log_probs(ref_policy, chosen_ids, chosen_mask)
                    ref_rejected_logps = self._get_log_probs(ref_policy, rejected_ids, rejected_mask)
                
                # DPO loss
                loss = self.dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps, ref_rejected_logps,
                )
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation
                loss.backward()
                
                if (global_step + 1) % self.config.gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.config.gradient_accumulation
                num_batches += 1
                global_step += 1
                
                # Save checkpoint
                if self.config.save_every > 0 and global_step % self.config.save_every == 0:
                    ckpt_path = output_dir / f"checkpoint-step-{global_step}.pth"
                    torch.save(policy.state_dict(), ckpt_path)
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"[DPO] Epoch {epoch+1}/{self.config.epochs} - Loss: {avg_loss:.4f} - Steps: {global_step}")
        
        # Save final model
        final_path = output_dir / "dpo_model.pth"
        torch.save(policy.state_dict(), final_path)
        policy.config.save_pretrained(str(output_dir))
        print(f"\n[DPO] Training complete! Saved to {output_dir}")
        
        return policy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO Training for Fusion-LLM")
    parser.add_argument("--model_path", type=str, default="output/mini_model")
    parser.add_argument("--data_path", type=str, default="data/preference_data.json")
    parser.add_argument("--output_path", type=str, default="output/dpo_model")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=256)
    
    args = parser.parse_args()
    
    config = DPOConfig(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_path,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    
    trainer = DPOTrainer(config)
    trainer.train()
