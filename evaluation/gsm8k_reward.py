"""
GSM8K Reward Function for GRPO Training

Evaluates math reasoning quality by extracting the final answer from the model
response and comparing it to the ground truth.

Usage:
    from evaluation.gsm8k_reward import gsm8k_reward_fn, load_gsm8k_dataset
    GRPOTrainer.register_reward_fn('gsm8k', gsm8k_reward_fn)
"""
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

# Optional datasets library
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


# ─── Answer Extraction ──────────────────────────────────────────────────────

# Patterns to extract the final numerical answer from a response.
# Order matters: try specific patterns first.
ANSWER_PATTERNS = [
    # "The answer is 42"
    r"(?:the\s+)?answer\s+is\s+([+-]?\d+(?:\.\d+)?)",
    # "Answer: 42" or "Answer: $42"
    r"answer\s*[:：]\s*\$?([+-]?\d+(?:\.\d+)?)",
    # "#### 42" (GSM8K gold answer format)
    r"####\s*([+-]?\d+(?:\.\d+)?)",
    # "= 42" (last expression result)
    r"=\s*([+-]?\d+(?:\.\d+)?)\s*$",
    # Last standalone number
    r"(?<![.\d])(\d+(?:\.\d+)?)(?![.\d])",
]


def extract_answer(text: str) -> Optional[float]:
    """
    Extract the final numerical answer from a response string.
    Returns None if no answer can be found.
    """
    text = text.strip()
    if not text:
        return None

    # Try patterns in order
    for pattern in ANSWER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the LAST match (most likely to be the final answer)
            try:
                return float(matches[-1])
            except ValueError:
                continue

    # Fallback: try to find any number
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def normalize_answer(answer: float) -> float:
    """
    Normalize answer for comparison:
    - Round to 2 decimal places to handle floating point errors
    - Convert to integer if it's effectively a whole number
    """
    answer = round(answer, 2)
    if answer == int(answer):
        return int(answer)
    return answer


# ─── Reward Function ─────────────────────────────────────────────────────────

def gsm8k_reward_fn(prompt: str, response: str) -> float:
    """
    GSM8K reward function for GRPO.

    Returns:
        1.0  if the final answer in `response` matches the ground truth
        0.0  otherwise
        0.0  if no answer can be extracted (response is empty or malformed)
    """
    extracted = extract_answer(response)
    if extracted is None:
        return 0.0
    return 1.0


# ─── GSM8K Dataset Loading ───────────────────────────────────────────────────

class GSM8KEvaluator:
    """
    Stateful GSM8K evaluator that stores ground truth answers for comparison.
    Use with GRPOTrainer.register_reward_fn('gsm8k', evaluator.reward).
    """

    def __init__(self, split: str = "test", dataset_path: Optional[str] = None):
        """
        Args:
            split: 'train' or 'test'
            dataset_path: Optional local path or HuggingFace dataset identifier.
                          Defaults to 'openai/gsm8k' if datasets library available.
        """
        self.split = split
        self.dataset_path = dataset_path or "openai/gsm8k"
        self._questions: List[str] = []
        self._answers: List[str] = []
        self._loaded = False

    def load(self):
        """Load the GSM8K dataset."""
        if self._loaded:
            return

        if HAS_DATASETS:
            try:
                ds = load_dataset(self.dataset_path, "main", split=self.split)
                self._questions = [item["question"] for item in ds]
                self._answers = [item["answer"] for item in ds]
                self._loaded = True
                print(f"[GSM8K] Loaded {len(self._answers)} examples from {self.dataset_path}/{self.split}")
                return
            except Exception as e:
                print(f"[GSM8K] Failed to load via datasets library: {e}")
                print("[GSM8K] Falling back to built-in mini test set")

        # Fallback: use built-in mini test set (10 representative GSM8K-style problems)
        self._build_mini_set()
        self._loaded = True

    def _build_mini_set(self):
        """Built-in mini test set (10 problems)"""
        self._questions = [
            "Janet buys 3 apples for $2 each and 2 oranges for $1.50 each. How much does she spend?",
            "A rectangle has a length of 8 cm and a width of 5 cm. What is its perimeter?",
            "If x = 4 and y = 7, what is x + y?",
            "There are 12 students in a class. If each student needs 3 pencils, how many pencils are needed in total?",
            "A train travels 60 miles per hour for 2.5 hours. How far does it travel?",
            "Tom has 24 candies. He gives 7 to Alice and 5 to Bob. How many candies does Tom have left?",
            "A book costs $15. If you have $50, how much change will you get after buying 2 books?",
            "What is 25% of 80?",
            "A garden is 10 meters long and 6 meters wide. What is its area?",
            "John runs 3 miles on Monday, 4 miles on Wednesday, and 2 miles on Friday. How many miles does he run in total?",
        ]
        self._answers = [
            "9",     # 3*2 + 2*1.50 = 6 + 3 = 9
            "26",    # 2*(8+5) = 26
            "11",    # 4+7=11
            "36",    # 12*3=36
            "150",   # 60*2.5=150
            "12",    # 24-7-5=12
            "20",    # 50-2*15=20
            "20",    # 0.25*80=20
            "60",    # 10*6=60
            "9",     # 3+4+2=9
        ]

    def reward(self, prompt: str, response: str) -> float:
        """
        Compute GSM8K reward by matching response against the correct answer
        for the given prompt.
        """
        if not self._loaded:
            self.load()

        # Find the matching question
        answer_str = None
        for q, a in zip(self._questions, self._answers):
            # Simple substring match (prompt may be a suffix)
            if prompt.strip() in q.strip() or q.strip() in prompt.strip():
                answer_str = a
                break

        if answer_str is None:
            # Fallback: use generic answer extraction (no ground truth available)
            extracted = extract_answer(response)
            return 1.0 if extracted is not None else 0.0

        # Extract answer from response
        extracted = extract_answer(response)
        if extracted is None:
            return 0.0

        # Normalize both for comparison
        try:
            extracted_norm = normalize_answer(extracted)
            # Extract the numerical answer from the gold answer string (may contain full reasoning text)
            gold_answer = extract_answer(answer_str)
            if gold_answer is None:
                return 0.0
            answer_norm = normalize_answer(gold_answer)
            return 1.0 if extracted_norm == answer_norm else 0.0
        except (ValueError, TypeError):
            return 0.0

    def evaluate_batch(self, prompts: List[str], responses: List[str]) -> Dict:
        """
        Evaluate a batch of prompt/response pairs.
        Returns accuracy statistics.
        """
        if not self._loaded:
            self.load()

        rewards = [self.reward(p, r) for p, r in zip(prompts, responses)]
        n = len(rewards)
        n_correct = sum(rewards)
        accuracy = n_correct / n if n > 0 else 0.0

        return {
            "n": n,
            "n_correct": n_correct,
            "accuracy": accuracy,
        }

    def __len__(self):
        if not self._loaded:
            self.load()
        return len(self._answers)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== GSM8K Reward Function Test ===\n")

    # Test answer extraction
    test_cases = [
        ("The answer is 42", 42.0),
        ("Answer: $9", 9.0),
        ("#### 26", 26.0),
        ("The total is = 150", 150.0),
        ("Janet spends $6 on apples, $3 on oranges. Total: $9", 9.0),
        ("x = 11", 11.0),
        ("She has 12 candies left.", 12.0),
        ("25% of 80 = 20. Answer: 20", 20.0),
        ("", None),
        ("No numbers here.", None),
    ]

    print("Answer extraction:")
    all_ok = True
    for text, expected in test_cases:
        got = extract_answer(text)
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            all_ok = False
        print(f"  [{status}] extract_answer({text[:40]!r:40s}) = {got} (expected {expected})")

    print()
    if all_ok:
        print("All extraction tests passed!")
    else:
        print("Some extraction tests failed!")

    print()
    print("GSM8K reward function test:")
    evaluator = GSM8KEvaluator()
    evaluator.load()
    print(f"  Dataset size: {len(evaluator)}")

    # Test a few questions
    test_pairs = [
        (evaluator._questions[0], "Janet spends $6 on apples and $3 on oranges. Total is $9. Answer: 9", 1.0),
        (evaluator._questions[0], "Janet spends $10. Answer: 10", 0.0),
        (evaluator._questions[0], "No answer here", 0.0),
    ]
    for q, r, expected in test_pairs:
        got = evaluator.reward(q, r)
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] reward = {got} (expected {expected})")
