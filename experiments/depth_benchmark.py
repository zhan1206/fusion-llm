"""
Depth Comparison Benchmark - ThinkingDial 核心验证实验

目标：验证 ThinkingDial 的不同 depth 在数学推理任务上的表现差异。
这是证明 ThinkingDial 机制有效性的关键实验。

实验设计：
1. 扩大模型配置到 ~10M 参数（hidden=512, layers=8, heads=8）
2. 用合成数学数据（四则运算）预训练所有 depth
3. 用 GRPO + GSM8K 奖励函数分别在不同 depth 下微调
4. 输出 depth 准确率对比表

Usage: python experiments/depth_benchmark.py
"""
import sys
import math
import random
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_model import FusionModel, FusionConfig
from models.thinking_dial import (
    ThinkingDialModel, ThinkingConfig, GRPOTrainer, GRPOConfig
)
from evaluation.gsm8k_reward import GSM8KEvaluator, extract_answer, normalize_answer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Benchmark] Device: {DEVICE}")
print()

# ─── 实验配置 ───────────────────────────────────────────────────────────────

EXP_CONFIG = {
    # ~5M 参数模型配置（CPU 可在合理时间内完成）
    "hidden_size": 256,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "intermediate_size": 512,
    "vocab_size": 1000,
    "max_position_embeddings": 128,
    "block_size": 32,
    "latent_dim": 16,
    # 训练参数
    "pretrain_epochs": 100,
    "pretrain_batch_size": 8,
    "pretrain_lr": 1e-3,
    "grpo_epochs": 30,
    "grpo_batch_size": 4,
    "grpo_lr": 1e-4,
    "grpo_num_samples": 3,
    "num_thinking_depths": 4,
    "test_batch_size": 50,
    "seed": 42,
}

def estimate_params(config: FusionConfig) -> int:
    """粗略估计模型参数量"""
    emb = config.vocab_size * config.hidden_size
    per_layer = (
        # attention: Q, K, V, O projections
        4 * config.hidden_size * config.hidden_size +
        # FFN: gate, up, down
        3 * config.hidden_size * config.intermediate_size +
        # SBLA latent projections (approx)
        2 * config.hidden_size * config.latent_dim +
        # norms
        2 * config.hidden_size
    )
    return emb + config.num_hidden_layers * per_layer + config.hidden_size * config.vocab_size

# ─── 合成数学数据 ──────────────────────────────────────────────────────────

def generate_math_data(n: int = 2000, ops: List[str] = None, max_val: int = 100,
                       seed: int = 42) -> List[Tuple[str, int, int, int]]:
    """
    生成合成数学运算数据。
    格式: (x, op_code, y, result)
    op_code: 0=+, 1=-, 2=*, 3=/
    """
    if ops is None:
        ops = ["+", "-"]  # 先只做加减法（乘除需要更大模型）
    op_map = {"+": 0, "-": 1, "*": 2, "/": 3}

    rng = random.Random(seed)
    data = []

    for _ in range(n):
        op = rng.choice(ops)
        code = op_map[op]

        if op == "+":
            x = rng.randint(1, max_val)
            y = rng.randint(1, max_val)
            result = x + y
        elif op == "-":
            x = rng.randint(1, max_val * 2)
            y = rng.randint(1, x)  # 保证结果 >= 0
            result = x - y
        elif op == "*":
            x = rng.randint(2, 20)
            y = rng.randint(2, 20)
            result = x * y
        elif op == "/":
            y = rng.randint(2, 20)
            result = rng.randint(1, 20)
            x = y * result  # 保证整除

        # 结果 clamp 到 vocab 范围
        result = max(0, min(result, EXP_CONFIG["vocab_size"] - 2))

        data.append((x, code, y, result))

    return data


def encode_example(x: int, op: int, y: int, result: int) -> Tuple[List[int], List[int]]:
    """
    编码数学题:
    input:  [2, x, op, y, 0, 0, 0]  (7 tokens, padded)
    labels: [-100, x, op, y, 99, result, 1]  (7 tokens, same length)
    Loss 计算: predict y->99, 99->result, result->1
    """
    input_ids = [2, x, op, y, 0, 0, 0]
    labels = [-100, x, op, y, 99, result, 1]
    return input_ids, labels


# ─── 梯度化奖励函数 ────────────────────────────────────────────────────────

def gradient_reward_fn(prompt: str, response: str) -> float:
    """
    梯度化奖励函数（用于合成数学任务）：
    - 能提取到数字 → +0.1
    - 数值接近正确值（误差 < 10%）→ +0.3
    - 完全正确 → +1.0
    """
    extracted = extract_answer(response)
    if extracted is None:
        return 0.0

    # 查找 ground truth
    gold = None
    try:
        # 从 prompt 中提取数字: CLS x OP y
        tokens = prompt.strip().split()
        if len(tokens) >= 3:
            x, op_code, y = int(tokens[0]), int(tokens[1]), int(tokens[2])
            if op_code == 0:
                gold = x + y
            elif op_code == 1:
                gold = x - y
            elif op_code == 2:
                gold = x * y
            elif op_code == 3:
                gold = x // y if y != 0 else 0
    except (ValueError, IndexError):
        pass

    if gold is None:
        return 0.1  # 至少输出了数字

    extracted_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(gold)

    if extracted_norm == gold_norm:
        return 1.0

    # 检查是否接近（误差 < 10%）
    if gold != 0:
        rel_error = abs(extracted_norm - gold_norm) / abs(gold_norm)
        if rel_error < 0.1:
            return 0.3

    # 输出了数字但答案错误
    return 0.1


# ─── 实验核心 ───────────────────────────────────────────────────────────────

def pretrain_model(config: FusionConfig, data: list) -> ThinkingDialModel:
    """阶段 1：用合成数学数据预训练"""
    model = FusionModel(config)
    thinking_config = ThinkingConfig(
        num_thinking_depths=EXP_CONFIG["num_thinking_depths"]
    )
    td_model = ThinkingDialModel(model, thinking_config)
    td_model.train().to(DEVICE)

    optimizer = torch.optim.AdamW(td_model.parameters(), lr=EXP_CONFIG["pretrain_lr"])

    losses = []
    for epoch in range(EXP_CONFIG["pretrain_epochs"]):
        batch = random.sample(data, min(EXP_CONFIG["pretrain_batch_size"], len(data)))

        input_batch = []
        target_batch = []
        for x, op, y, result in batch:
            inp, tgt = encode_example(x, op, y, result)
            input_batch.append(inp)
            target_batch.append(tgt)

        input_ids = torch.tensor(input_batch, device=DEVICE, dtype=torch.long)
        labels = torch.tensor(target_batch, device=DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = td_model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(td_model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            avg = sum(losses[-10:]) / 10
            print(f"    Pretrain Epoch {epoch+1}/{EXP_CONFIG['pretrain_epochs']}: "
                  f"Loss = {avg:.4f}")

    final_loss = sum(losses[-10:]) / 10
    print(f"  Pretrain complete. Final loss: {final_loss:.4f}")
    return td_model


def grpo_finetune(td_model: ThinkingDialModel, data: list,
                  depth: int, gsm8k_eval: GSM8KEvaluator = None) -> Dict:
    """
    阶段 2：GRPO 微调（使用合成数学数据 + 可选 GSM8K 奖励）

    Returns:
        {losses, rewards, depth}
    """
    grpo_config = GRPOConfig(
        grpo_sample_size=EXP_CONFIG["grpo_num_samples"],
        kl_coef=0.05,
    )

    # 设置奖励函数
    def reward_fn(prompt, response):
        if gsm8k_eval is not None:
            return gsm8k_eval.reward(prompt, response)
        return gradient_reward_fn(prompt, response)

    td_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in td_model.parameters() if p.requires_grad],
        lr=EXP_CONFIG["grpo_lr"]
    )

    epoch_losses = []
    epoch_rewards = []

    for epoch in range(EXP_CONFIG["grpo_epochs"]):
        batch = random.sample(data, min(EXP_CONFIG["grpo_batch_size"], len(data)))
        batch_losses = []
        batch_rewards = []

        for x, op, y, result in batch:
            inp, _ = encode_example(x, op, y, result)
            input_ids = torch.tensor([inp], device=DEVICE, dtype=torch.long)

            # 用指定 depth 生成多个样本
            gen_ids = []
            for _ in range(EXP_CONFIG["grpo_num_samples"]):
                with torch.no_grad():
                    out = td_model.generate(
                        input_ids,
                        max_new_tokens=8,
                        thinking_depth=depth,
                        do_sample=True,
                        temperature=1.0,
                        pad_token_id=0,
                    )
                gen_ids.append(out[0].tolist())

            # 计算奖励
            rewards = []
            for gen in gen_ids:
                new_tokens = gen[len(inp):]
                # Extract result: expected format [99, result, 1, ...]
                if len(new_tokens) >= 2 and new_tokens[0] == 99:
                    predicted = new_tokens[1]
                else:
                    predicted = None

                # Compute reward
                if predicted is not None and predicted == result:
                    r = 1.0
                elif predicted is not None:
                    # Gradient reward: check proximity
                    if result != 0 and abs(predicted - result) / abs(result) < 0.1:
                        r = 0.3
                    else:
                        r = 0.1
                else:
                    r = 0.0
                rewards.append(r)

            # GRPO: 基于奖励计算 loss
            if len(rewards) > 1 and sum(rewards) > 0:
                # 归一化奖励
                rewards_t = torch.tensor(rewards, device=DEVICE, dtype=torch.float)
                mean_r = rewards_t.mean()
                std_r = rewards_t.std() + 1e-8
                advantages = (rewards_t - mean_r) / std_r

                # 对每个样本计算 policy gradient loss
                total_loss = torch.tensor(0.0, device=DEVICE)
                for i, gen in enumerate(gen_ids):
                    gen_t = torch.tensor([gen], device=DEVICE, dtype=torch.long)
                    # Mask labels: only compute loss on generated tokens
                    gen_labels = gen_t[:, 1:].clone()
                    gen_labels[:, :len(inp)-1] = -100  # ignore input portion
                    # Forward to get log probs
                    outputs = td_model(gen_t[:, :-1], labels=gen_labels)
                    neg_log_prob = outputs.loss
                    # Policy gradient: -advantage * log_prob
                    total_loss += advantages[i] * neg_log_prob

                loss = total_loss / len(gen_ids)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(td_model.parameters(), 1.0)
                optimizer.step()

                batch_losses.append(loss.item())
                batch_rewards.append(mean_r.item())
            else:
                batch_losses.append(0.0)
                batch_rewards.append(0.0)

        epoch_losses.append(sum(batch_losses) / len(batch_losses))
        epoch_rewards.append(sum(batch_rewards) / len(batch_rewards))

        if (epoch + 1) % 10 == 0:
            print(f"    GRPO Epoch {epoch+1}/{EXP_CONFIG['grpo_epochs']}: "
                  f"Loss = {epoch_losses[-1]:.4f}, "
                  f"Mean Reward = {epoch_rewards[-1]:.4f}")

    print(f"  GRPO complete. Final reward: {epoch_rewards[-1]:.4f}")
    return {
        "losses": epoch_losses,
        "rewards": epoch_rewards,
        "depth": depth,
    }


def evaluate_accuracy(td_model: ThinkingDialModel, data: list,
                      depth: int, n_samples: int = None) -> Dict:
    """
    在测试集上评估指定 depth 的准确率。
    期望生成格式: [..., 99, result, 1, ...] -> 取 position 5 的 token 作为结果
    """
    if n_samples is None:
        n_samples = EXP_CONFIG["test_batch_size"]

    test_data = random.sample(data, min(n_samples, len(data)))
    correct = 0
    total = len(test_data)
    results_by_op = {"+": {"correct": 0, "total": 0},
                     "-": {"correct": 0, "total": 0},
                     "*": {"correct": 0, "total": 0},
                     "/": {"correct": 0, "total": 0}}

    td_model.eval()
    op_names = {0: "+", 1: "-", 2: "*", 3: "/"}

    for x, op, y, result in test_data:
        inp, _ = encode_example(x, op, y, result)
        input_ids = torch.tensor([inp], device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            out = td_model.generate(
                input_ids,
                max_new_tokens=8,
                thinking_depth=depth,
                do_sample=False,
                pad_token_id=0,
            )

        gen_tokens = out[0, len(inp):].tolist()
        # Expected: model generates [99, result, 1, ...]
        gen_result = None
        if len(gen_tokens) >= 2 and gen_tokens[0] == 99:
            gen_result = gen_tokens[1]
        elif gen_tokens:
            gen_result = gen_tokens[0]

        op_name = op_names.get(op, "?")
        results_by_op[op_name]["total"] += 1

        if gen_result is not None:
            if normalize_answer(gen_result) == normalize_answer(result):
                correct += 1
                results_by_op[op_name]["correct"] += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "depth": depth,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_op": results_by_op,
    }


# ─── 消融实验 ───────────────────────────────────────────────────────────────

def create_vanilla_model(config: FusionConfig):
    """
    创建无 ThinkingDial 的 vanilla Transformer 对照组。
    直接用 FusionModel（无 thinking depth 偏置）。
    """
    model = FusionModel(config)
    model.train().to(DEVICE)
    return model


def pretrain_vanilla(model: FusionModel, data: list) -> Dict:
    """Vanilla Transformer 预训练（无 ThinkingDial）"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=EXP_CONFIG["pretrain_lr"])

    losses = []
    for epoch in range(EXP_CONFIG["pretrain_epochs"]):
        batch = random.sample(data, min(EXP_CONFIG["pretrain_batch_size"], len(data)))

        input_batch = []
        target_batch = []
        for x, op, y, result in batch:
            inp, tgt = encode_example(x, op, y, result)
            input_batch.append(inp)
            target_batch.append(tgt)

        input_ids = torch.tensor(input_batch, device=DEVICE, dtype=torch.long)
        labels = torch.tensor(target_batch, device=DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

    final_loss = sum(losses[-10:]) / 10
    return {"losses": losses, "final_loss": final_loss}


def evaluate_vanilla(model: FusionModel, data: list, n_samples: int = None) -> Dict:
    """评估 vanilla Transformer 准确率"""
    if n_samples is None:
        n_samples = EXP_CONFIG["test_batch_size"]

    test_data = random.sample(data, min(n_samples, len(data)))
    correct = 0
    total = len(test_data)

    model.eval()

    for x, op, y, result in test_data:
        inp, _ = encode_example(x, op, y, result)
        input_ids = torch.tensor([inp], device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=0,
            )

        gen_tokens = out[0, len(inp):].tolist()
        gen_result = None
        if len(gen_tokens) >= 2 and gen_tokens[0] == 99:
            gen_result = gen_tokens[1]
        else:
            gen_result = extract_answer(" ".join(str(t) for t in gen_tokens))

        if gen_result is not None:
            if normalize_answer(gen_result) == normalize_answer(result):
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}


# ─── 主实验流程 ──────────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 70)
    print("ThinkingDial Depth Comparison Benchmark")
    print("=" * 70)
    print()

    random.seed(EXP_CONFIG["seed"])
    torch.manual_seed(EXP_CONFIG["seed"])

    # 配置模型
    config = FusionConfig(
        vocab_size=EXP_CONFIG["vocab_size"],
        hidden_size=EXP_CONFIG["hidden_size"],
        num_hidden_layers=EXP_CONFIG["num_hidden_layers"],
        num_attention_heads=EXP_CONFIG["num_attention_heads"],
        intermediate_size=EXP_CONFIG["intermediate_size"],
        max_position_embeddings=EXP_CONFIG["max_position_embeddings"],
        block_size=EXP_CONFIG["block_size"],
        latent_dim=EXP_CONFIG["latent_dim"],
    )

    n_params = estimate_params(config)
    print(f"Model config: {EXP_CONFIG['hidden_size']}d x {EXP_CONFIG['num_hidden_layers']}L x {EXP_CONFIG['num_attention_heads']}H")
    print(f"Estimated parameters: ~{n_params:,}")
    print()

    # 生成训练数据
    train_data = generate_math_data(n=500, ops=["+"], max_val=20, seed=42)
    test_data = generate_math_data(n=100, ops=["+"], max_val=20, seed=999)
    print(f"Training set: {len(train_data)} examples")
    print(f"Test set: {len(test_data)} examples")
    print()

    # ─── 实验 1: Depth 对比（相同模型，不同 depth） ─────────────────────
    print("=" * 60)
    print("EXPERIMENT 1: Depth Comparison (Same Model, Different Depths)")
    print("=" * 60)
    print()

    t0 = time.time()
    print("[1/4] Pretraining ThinkingDial model...")
    td_model = pretrain_model(config, train_data)
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    # 在预训练后立即评估所有 depth（零样本）
    print("[2/4] Zero-shot accuracy by depth (before GRPO)...")
    zero_shot_results = {}
    for depth in range(EXP_CONFIG["num_thinking_depths"]):
        result = evaluate_accuracy(td_model, test_data, depth)
        zero_shot_results[depth] = result
        print(f"  Depth {depth}: {result['accuracy']*100:.1f}% "
              f"({result['correct']}/{result['total']})")
    print()

    # 对每个 depth 进行 GRPO 微调（共享预训练权重）
    print("[3/4] GRPO fine-tuning per depth...")
    grpo_results = {}
    for depth in range(EXP_CONFIG["num_thinking_depths"]):
        print(f"  --- Depth {depth} ---")
        # 每次从预训练检查点开始
        td_model_copy = pretrain_model(config, train_data)  # 重新预训练保证公平
        grpo_res = grpo_finetune(td_model_copy, train_data, depth)
        grpo_results[depth] = grpo_res

        # 评估
        result = evaluate_accuracy(td_model_copy, test_data, depth)
        grpo_results[depth]["eval"] = result
        print(f"  Post-GRPO Accuracy: {result['accuracy']*100:.1f}%")
        print()

        # 清理
        del td_model_copy
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print("[4/4] Post-GRPO accuracy by depth...")
    post_grpo_results = {}
    for depth in range(EXP_CONFIG["num_thinking_depths"]):
        result = grpo_results[depth]["eval"]
        post_grpo_results[depth] = result
        print(f"  Depth {depth}: {result['accuracy']*100:.1f}% "
              f"({result['correct']}/{result['total']})")
    print()

    # ─── 实验 2: 消融实验（Vanilla vs ThinkingDial） ──────────────────
    print("=" * 60)
    print("EXPERIMENT 2: Ablation (Vanilla Transformer vs ThinkingDial)")
    print("=" * 60)
    print()

    print("[1/2] Pretraining vanilla Transformer...")
    t1 = time.time()
    vanilla_model = create_vanilla_model(config)
    vanilla_train = pretrain_vanilla(vanilla_model, train_data)
    print(f"  Pretrain loss: {vanilla_train['final_loss']:.4f}")
    print(f"  Time: {time.time() - t1:.1f}s")
    print()

    print("[2/2] Evaluating vanilla Transformer...")
    vanilla_result = evaluate_vanilla(vanilla_model, test_data)
    print(f"  Vanilla accuracy: {vanilla_result['accuracy']*100:.1f}% "
          f"({vanilla_result['correct']}/{vanilla_result['total']})")
    print()

    # ─── 结果汇总 ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    print("Table 1: Zero-shot Accuracy by Thinking Depth")
    print("-" * 50)
    print(f"{'Depth':<10} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 50)
    for depth in range(EXP_CONFIG["num_thinking_depths"]):
        r = zero_shot_results[depth]
        print(f"{depth:<10} {r['accuracy']*100:>6.1f}%       {r['correct']}/{r['total']}")
    print()

    print("Table 2: Post-GRPO Accuracy by Thinking Depth")
    print("-" * 50)
    print(f"{'Depth':<10} {'Accuracy':<12} {'Correct/Total':<15} {'Final Reward':<15}")
    print("-" * 50)
    for depth in range(EXP_CONFIG["num_thinking_depths"]):
        r = grpo_results[depth]["eval"]
        reward = grpo_results[depth]["rewards"][-1]
        print(f"{depth:<10} {r['accuracy']*100:>6.1f}%       "
              f"{r['correct']}/{r['total']:<10} {reward:>8.4f}")
    print()

    print("Table 3: Ablation Comparison")
    print("-" * 50)
    print(f"{'Model':<25} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 50)
    # ThinkingDial best depth
    best_depth = max(range(EXP_CONFIG["num_thinking_depths"]),
                     key=lambda d: post_grpo_results[d]["accuracy"])
    td_acc = post_grpo_results[best_depth]
    print(f"ThinkingDial (depth={best_depth})   {td_acc['accuracy']*100:>6.1f}%       "
          f"{td_acc['correct']}/{td_acc['total']}")
    print(f"Vanilla Transformer          {vanilla_result['accuracy']*100:>6.1f}%       "
          f"{vanilla_result['correct']}/{vanilla_result['total']}")
    print()

    # 分析
    td_best_acc = post_grpo_results[best_depth]["accuracy"]
    vanilla_acc = vanilla_result["accuracy"]

    print("Analysis:")
    if td_best_acc > vanilla_acc:
        improvement = (td_best_acc - vanilla_acc) / vanilla_acc * 100
        print(f"  ThinkingDial (depth={best_depth}) outperforms vanilla by +{improvement:.1f}%")
    elif td_best_acc == vanilla_acc:
        print(f"  ThinkingDial matches vanilla performance")
    else:
        degradation = (vanilla_acc - td_best_acc) / vanilla_acc * 100
        print(f"  ThinkingDial underperforms vanilla by -{degradation:.1f}%")
        print(f"  Note: This may indicate architecture needs adjustment or training is insufficient")

    # Depth variance analysis
    accuracies = [post_grpo_results[d]["accuracy"] for d in range(EXP_CONFIG["num_thinking_depths"])]
    if max(accuracies) > 0 and min(accuracies) < max(accuracies):
        print(f"  Depth sensitivity: accuracy ranges from "
              f"{min(accuracies)*100:.1f}% (depth {accuracies.index(min(accuracies))}) "
              f"to {max(accuracies)*100:.1f}% (depth {accuracies.index(max(accuracies))})")
        print(f"  ThinkingDial produces different outcomes at different depths")

    print()
    total_time = time.time() - t0
    print(f"Total benchmark time: {total_time:.1f}s")
    print()

    return {
        "zero_shot": zero_shot_results,
        "post_grpo": post_grpo_results,
        "grpo": grpo_results,
        "vanilla": vanilla_result,
    }


if __name__ == "__main__":
    results = run_benchmark()
