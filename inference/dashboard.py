#!/usr/bin/env python3
"""
Fusion Inference Dashboard - Interactive inference control panel

Provides real-time control over:
1. Thinking Dial intensity (think_rank)
2. Temperature / top-p / top-k sampling
3. Max generation length
4. SBLA mode (pure_sbla / hybrid)
5. Streaming output

Usage:
    python inference/dashboard.py --model_path output/mini_model
    python inference/dashboard.py --model_path output/mini_model --web --port 7860

Author: Zhu Zizhan
Project: Fusion-LLM
License: Apache 2.0
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.fusion_model import FusionModel, FusionConfig


@dataclass
class InferenceConfig:
    """Inference-time configuration."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256
    think_rank: int = 0          # 0=fast, 1=normal, 2=deep, 3=deepest
    repetition_penalty: float = 1.1
    do_sample: bool = True
    sbla_mode: str = "hybrid"    # pure_sbla / hybrid


class InferenceEngine:
    """
    Fusion model inference engine with Thinking Dial control.
    
    The Thinking Dial adjusts generation behavior based on think_rank:
    - Rank 0: Fast mode - lower temperature, shorter output
    - Rank 1: Normal mode - standard settings
    - Rank 2: Deep mode - higher temperature, longer output, more exploration
    - Rank 3: Deepest mode - highest temperature, max exploration, chain-of-thought
    """
    
    THINK_RANK_PRESETS = {
        0: {"temperature": 0.3, "top_p": 0.85, "max_new_tokens": 128, "repetition_penalty": 1.2},
        1: {"temperature": 0.7, "top_p": 0.90, "max_new_tokens": 256, "repetition_penalty": 1.1},
        2: {"temperature": 0.9, "top_p": 0.95, "max_new_tokens": 512, "repetition_penalty": 1.05},
        3: {"temperature": 1.0, "top_p": 0.98, "max_new_tokens": 1024, "repetition_penalty": 1.0},
    }
    
    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model, self.config = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.inference_config = InferenceConfig()
        self.kv_cache = None
        
    def _load_model(self, model_path: str) -> tuple:
        """Load model from path."""
        config_path = Path(model_path)
        
        if config_path.joinpath("config.json").exists():
            model_config = FusionConfig.from_pretrained(str(config_path))
        else:
            model_config = FusionConfig(
                vocab_size=10000, hidden_size=256, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=512,
                block_size=64, latent_dim=16, max_position_embeddings=256,
            )
        
        model = FusionModel(model_config)
        
        weight_path = config_path / "final_model.pth"
        if not weight_path.exists():
            weight_path = config_path / "dpo_model.pth"
        
        if weight_path.exists():
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {weight_path}")
        else:
            print(f"Warning: No weights found at {config_path}, using random init")
        
        return model, model_config
    
    def set_think_rank(self, rank: int):
        """Apply Thinking Dial preset by rank."""
        if rank not in self.THINK_RANK_PRESETS:
            print(f"Invalid think_rank {rank}, must be 0-3")
            return
        
        preset = self.THINK_RANK_PRESETS[rank]
        self.inference_config.think_rank = rank
        self.inference_config.temperature = preset["temperature"]
        self.inference_config.top_p = preset["top_p"]
        self.inference_config.max_new_tokens = preset["max_new_tokens"]
        self.inference_config.repetition_penalty = preset["repetition_penalty"]
        print(f"[Thinking Dial] Rank {rank}: temp={preset['temperature']}, "
              f"top_p={preset['top_p']}, max_tokens={preset['max_new_tokens']}")
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization."""
        encoded = list(text.encode('utf-8'))[:self.config.max_position_embeddings]
        return torch.tensor([encoded], dtype=torch.long).to(self.device)
    
    def _detokenize(self, token_ids: list) -> str:
        """Convert token IDs back to text."""
        try:
            return bytes(token_ids).decode('utf-8', errors='replace')
        except Exception:
            return ""
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            config: Override inference config
            stream: If True, yield tokens one by one
            
        Returns:
            Generated text (or generator if stream=True)
        """
        cfg = config or self.inference_config
        input_ids = self._tokenize(prompt)
        
        if stream:
            return self._generate_stream(input_ids, cfg)
        
        # Full generation
        generated = input_ids[0].tolist()
        self.kv_cache = None
        
        for _ in range(cfg.max_new_tokens):
            input_tensor = input_ids if self.kv_cache is None else input_ids[:, -1:]
            attention_mask = (input_ids != 0).float()
            
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=self.kv_cache,
            )
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs['logits'][:, -1, :]
            
            # Apply repetition penalty
            if cfg.repetition_penalty != 1.0:
                for token_id in set(generated[-50:]):
                    logits[0, token_id] /= cfg.repetition_penalty
            
            # Temperature
            logits = logits / cfg.temperature
            
            # Top-k filtering
            if cfg.top_k > 0:
                top_k = min(cfg.top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1:]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if cfg.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > cfg.top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            if cfg.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            next_id = next_token.item()
            generated.append(next_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Update KV cache
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                self.kv_cache = outputs.past_key_values
        
        # Decode only new tokens
        new_tokens = generated[len(self._tokenize(prompt)[0]):]
        return self._detokenize(new_tokens)
    
    def _generate_stream(self, input_ids, cfg) -> Generator:
        """Streaming generation."""
        generated = input_ids[0].tolist()
        self.kv_cache = None
        
        for _ in range(cfg.max_new_tokens):
            input_tensor = input_ids if self.kv_cache is None else input_ids[:, -1:]
            attention_mask = (input_ids != 0).float()
            
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=self.kv_cache,
            )
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits[:, -1, :]
            else:
                logits = outputs['logits'][:, -1, :]
            
            logits = logits / cfg.temperature
            
            if cfg.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            next_id = next_token.item()
            generated.append(next_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
                self.kv_cache = outputs.past_key_values
            
            text = self._detokenize([next_id])
            if text:
                yield text


def interactive_mode(engine: InferenceEngine):
    """Run interactive inference session."""
    print("\n" + "=" * 60)
    print("Fusion Inference Dashboard - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  /think <0-3>    - Set Thinking Dial rank")
    print("  /temp <float>   - Set temperature")
    print("  /topp <float>   - Set top-p")
    print("  /topk <int>     - Set top-k")
    print("  /maxlen <int>   - Set max new tokens")
    print("  /config         - Show current config")
    print("  /quit           - Exit")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input == "/quit":
            break
        elif user_input == "/config":
            cfg = engine.inference_config
            print(f"  think_rank={cfg.think_rank}, temp={cfg.temperature}, "
                  f"top_p={cfg.top_p}, top_k={cfg.top_k}, max_tokens={cfg.max_new_tokens}")
            continue
        elif user_input.startswith("/think "):
            rank = int(user_input.split()[1])
            engine.set_think_rank(rank)
            continue
        elif user_input.startswith("/temp "):
            engine.inference_config.temperature = float(user_input.split()[1])
            print(f"  Temperature set to {engine.inference_config.temperature}")
            continue
        elif user_input.startswith("/topp "):
            engine.inference_config.top_p = float(user_input.split()[1])
            print(f"  Top-p set to {engine.inference_config.top_p}")
            continue
        elif user_input.startswith("/topk "):
            engine.inference_config.top_k = int(user_input.split()[1])
            print(f"  Top-k set to {engine.inference_config.top_k}")
            continue
        elif user_input.startswith("/maxlen "):
            engine.inference_config.max_new_tokens = int(user_input.split()[1])
            print(f"  Max length set to {engine.inference_config.max_new_tokens}")
            continue
        
        # Generate
        start_time = time.time()
        response = engine.generate(user_input)
        elapsed = time.time() - start_time
        
        print(f"\nFusion [{engine.inference_config.think_rank}]: {response}")
        print(f"  ({elapsed:.2f}s)")


def main():
    parser = argparse.ArgumentParser(description="Fusion Inference Dashboard")
    parser.add_argument("--model_path", type=str, default="output/mini_model")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--think_rank", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--web", action="store_true", help="Launch web interface (requires gradio)")
    parser.add_argument("--port", type=int, default=7860)
    
    args = parser.parse_args()
    
    engine = InferenceEngine(args.model_path, args.device)
    engine.set_think_rank(args.think_rank)
    
    if args.temperature is not None:
        engine.inference_config.temperature = args.temperature
    
    if args.web:
        try:
            import gradio as gr
            
            def generate_response(prompt, think_rank, temperature, top_p, max_tokens):
                engine.set_think_rank(int(think_rank))
                engine.inference_config.temperature = temperature
                engine.inference_config.top_p = top_p
                engine.inference_config.max_new_tokens = int(max_tokens)
                return engine.generate(prompt)
            
            demo = gr.Interface(
                fn=generate_response,
                inputs=[
                    gr.Textbox(label="Prompt", lines=3),
                    gr.Slider(0, 3, step=1, value=0, label="Think Rank"),
                    gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
                    gr.Slider(0.5, 1.0, value=0.9, label="Top-p"),
                    gr.Slider(64, 2048, step=64, value=256, label="Max Tokens"),
                ],
                outputs=gr.Textbox(label="Response", lines=10),
                title="Fusion Inference Dashboard",
                description="Control Thinking Dial intensity and generation parameters",
            )
            demo.launch(server_port=args.port)
        except ImportError:
            print("Gradio not installed. Install: pip install gradio")
            print("Falling back to interactive mode...")
            interactive_mode(engine)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()
