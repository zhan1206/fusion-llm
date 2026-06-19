---
title: Fusion-LLM Demo
emoji: 🧠
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: apache-2.0
---

# Fusion-LLM: Train & Chat Demo

A compact, interpretable Chinese LLM framework featuring **SBLA (Sliding Block Latent Attention)** and **Thinking Dial** dynamic inference control.

## What This Space Does

1. **Train** a FusionMini model on Chinese AI Q&A pairs — configure model size, layers, epochs
2. **Chat** with the trained model — ask questions and get generated responses

## Quick Start

1. Click the **"Train Model"** tab
2. Adjust parameters (defaults work great)
3. Click **"Start Training"** — watch loss decrease in real-time
4. Switch to **"Chat"** tab and type a question!

## Architecture

| Component | Description |
|-----------|-------------|
| **SBLA Attention** | Block-level latent attention with gated merging |
| **Thinking Dial** | Dynamic inference depth via `<|think_depth_N|>` tokens |
| **FusionMini** | Lightweight proof-of-concept (~3M params, trains in seconds) |

## Links

- [GitHub Repository](https://github.com/zhan1206/fusion-llm)
- [Full Documentation](https://github.com/zhan1206/fusion-llm/tree/master/docs)
