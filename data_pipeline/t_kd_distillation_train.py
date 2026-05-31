"""
T-KD 蒸馏训练脚本

真实的蒸馏训练逻辑：
1. 加载教师模型（冻结参数）
2. 加载学生模型（可训练）
3. 计算 KL 散度损失（教师 logits vs 学生 logits）
4. 可选：加入硬标签交叉熵损失

使用方法：
    python data_pipeline/t_kd_distillation_train.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --student_model "./output/fusion-mini" \
        --train_data "data/t_kd_corpus.jsonl" \
        --output_dir "./output/fusion-mini-distilled"

作者：zhan1206
项目：Fusion - 六边形开源大模型
许可证：Apache 2.0
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationDataset(Dataset):
    """蒸馏训练数据集"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"✅ 加载数据：{len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 编码
        text = item.get("text", "")
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # labels（用于交叉熵损失）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DistillationTrainer:
    """
    T-KD 蒸馏训练器
    
    核心：学生模型模仿教师模型的输出分布
    """
    
    def __init__(
        self,
        teacher_model_name: str,
        student_model_name: str,
        device: str = "cuda",
        temperature: float = 4.0,
        alpha: float = 0.5,  # KL 损失权重
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        grad_accum_steps: int = 8,
    ):
        """
        初始化蒸馏训练器
        
        参数：
            teacher_model_name: 教师模型（HuggingFace ID 或本地路径）
            student_model_name: 学生模型（本地路径，将训练）
            device: 设备
            temperature: 蒸馏温度（T>1 软化概率分布）
            alpha: 损失权重（alpha * KL + (1-alpha) * CE）
            learning_rate: 学习率
            batch_size: 批次大小
            grad_accum_steps: 梯度累积步数
        """
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        
        # 1. 加载教师模型（冻结）
        logger.info(f"📚 加载教师模型：{teacher_model_name}")
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_model_name,
            trust_remote_code=True,
        )
        
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info(f"✅ 教师模型加载完成（参数已冻结）")
        
        # 2. 加载学生模型（可训练）
        logger.info(f"🎓 加载学生模型：{student_model_name}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            student_model_name,
            trust_remote_code=True,
        )
        
        self.student_model = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        
        self.student_model.train()
        
        logger.info(f"✅ 学生模型加载完成（可训练）")
        
        # 3. 优化器 + 学习率调度器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        logger.info(f"✅ 优化器初始化完成（lr={learning_rate}）")
    
    def compute_distillation_loss(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算蒸馏损失
        
        公式：Loss = alpha * T² * KL(teacher || student) + (1-alpha) * CE(student, labels)
        
        参数：
            teacher_logits: (batch, seq_len, vocab_size)
            student_logits: (batch, seq_len, vocab_size)
            labels: (batch, seq_len)
            
        返回：
            蒸馏损失
        """
        # 1. KL 散度损失（蒸馏）
        T = self.temperature
        T_squared = T * T
        
        # 软化概率分布
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        
        # KL 散度（教师 || 学生）
        kl_loss = F.kl_div(
            student_log_probs.view(-1, student_logits.size(-1)),
            teacher_probs.view(-1, teacher_logits.size(-1)),
            reduction="batchmean",
            log_target=False,
        ) * T_squared  # 温度缩放
        
        # 2. 交叉熵损失（硬标签）
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        # 3. 总损失
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train(
        self,
        train_data_path: str,
        output_dir: str,
        num_epochs: int = 3,
        save_steps: int = 500,
        max_length: int = 2048,
    ):
        """
        执行蒸馏训练
        
        参数：
            train_data_path: 训练数据路径（.jsonl）
            output_dir: 模型保存目录
            num_epochs: 训练轮数
            save_steps: 每 N 步保存一次
            max_length: 最大序列长度
        """
        # 1. 创建数据集
        train_dataset = DistillationDataset(
            train_data_path,
            self.student_tokenizer,
            max_length=max_length,
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows 下设为 0
        )
        
        # 2. 学习率调度器
        total_steps = len(train_dataloader) * num_epochs // self.grad_accum_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )
        
        # 3. 训练循环
        logger.info(f"🚀 开始蒸馏训练...")
        logger.info(f"   轮数：{num_epochs}")
        logger.info(f"   批次大小：{self.batch_size}")
        logger.info(f"   梯度累积：{self.grad_accum_steps}")
        logger.info(f"   温度：{self.temperature}")
        logger.info(f"   Alpha：{self.alpha}")
        
        global_step = 0
        self.student_model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_ce = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # 移动到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 教师模型推理（无梯度）
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    teacher_logits = teacher_outputs.logits
                
                # 学生模型前向传播
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs.logits
                
                # 计算蒸馏损失
                loss, kl_loss, ce_loss = self.compute_distillation_loss(
                    teacher_logits,
                    student_logits,
                    labels,
                )
                
                # 梯度累积（除以累积步数）
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    max_norm=1.0,
                )
                
                # 更新参数（每 grad_accum_steps 步）
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # 统计
                epoch_loss += loss.item() * self.grad_accum_steps
                epoch_kl += kl_loss.item()
                epoch_ce += ce_loss.item()
                num_batches += 1
                
                # 日志
                if batch_idx % 10 == 0:
                    logger.info(
                        f"   Epoch {epoch+1}, Batch {batch_idx}, "
                        f"Loss: {loss.item() * self.grad_accum_steps:.4f}, "
                        f"KL: {kl_loss.item():.4f}, CE: {ce_loss.item():.4f}"
                    )
                
                # 清理 GPU 缓存
                del teacher_logits, student_logits, loss
                torch.cuda.empty_cache()
            
            # Epoch 结束统计
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_kl = epoch_kl / max(num_batches, 1)
            avg_ce = epoch_ce / max(num_batches, 1)
            
            logger.info(f"   Epoch {epoch+1}/{num_epochs} 完成")
            logger.info(f"   Average Loss: {avg_loss:.4f}")
            logger.info(f"   Average KL: {avg_kl:.4f}")
            logger.info(f"   Average CE: {avg_ce:.4f}")
            
            # 保存检查点
            checkpoint_dir = Path(output_dir) / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            self.student_model.save_pretrained(checkpoint_dir)
            self.student_tokenizer.save_pretrained(checkpoint_dir)
            
            logger.info(f"   ✅ 检查点保存至：{checkpoint_dir}")
        
        # 4. 保存最终模型
        output_path = Path(output_dir) / "final"
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.student_model.save_pretrained(output_path)
        self.student_tokenizer.save_pretrained(output_path)
        
        logger.info(f"🎉 蒸馏训练完成！模型保存至：{output_path}")
    
    def evaluate(
        self,
        eval_data_path: str,
        max_length: int = 2048,
        num_samples: int = 100,
    ):
        """
        评估蒸馏后的模型
        
        参数：
            eval_data_path: 评估数据路径
            max_length: 最大序列长度
            num_samples: 评估样本数
        """
        logger.info(f"📊 开始评估...")
        
        self.student_model.eval()
        
        # 加载评估数据
        eval_dataset = DistillationDataset(
            eval_data_path,
            self.student_tokenizer,
            max_length=max_length,
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                if num_batches >= num_samples:
                    break
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        logger.info(f"✅ 评估完成")
        logger.info(f"   Average Loss: {avg_loss:.4f}")
        logger.info(f"   Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.2f}")


def main():
    parser = argparse.ArgumentParser(description="T-KD 蒸馏训练")
    
    parser.add_argument(
        "--teacher_model",
        type=str,
        required=True,
        help="教师模型（HuggingFace ID 或本地路径）",
    )
    
    parser.add_argument(
        "--student_model",
        type=str,
        required=True,
        help="学生模型（本地路径，将训练）",
    )
    
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="训练数据路径（.jsonl）",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="模型保存目录",
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="训练轮数",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="批次大小",
    )
    
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="蒸馏温度（T>1 软化概率分布）",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="损失权重（alpha * KL + (1-alpha) * CE）",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="最大序列长度",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda/cpu）",
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化训练器
    trainer = DistillationTrainer(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        device=args.device,
        temperature=args.temperature,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
    )
    
    # 训练
    trainer.train(
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
    )
    
    logger.info("🎉 蒸馏训练完成！")


if __name__ == "__main__":
    main()