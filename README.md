---
title: Fusion-LLM Demo
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
python_version: 3.11
app_file: app.py
pinned: false
license: apache-2.0
---

# Fusion-LLM Demo

Train and chat with **Fusion-LLM** — a custom Transformer with SBLA (Sparse Block Latent Attention) and Thinking Dial.

## Features

- **Train**: Configure hyperparameters and train the model in-browser
- **Chat**: Interact with your trained model
- **About**: Learn about the architecture (SBLA + Thinking Dial)

## Architecture

- **SBLA Attention**: Block-level sparse latent attention with gated merging
- **Thinking Dial**: Configurable reasoning depth via `<|think_depth_N|>` tokens
- **FusionMini**: ~3M parameter model for fast CPU training
