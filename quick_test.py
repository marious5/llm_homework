"""Quick smoke test for task1_gsm8k and task2_scienceqa.

This script only runs a very small number of samples (default: 3) for each
 task to verify that the environment, dependencies, and model pipelines
 are working correctly.

Run:
    python quick_test.py
"""

import torch

from task1_gsm8k import GSM8KEvaluator
from task2_scienceqa import ScienceQAEvaluator


def test_task1(num_samples: int = 3) -> None:
    print("\n===== Quick Test: Task 1 (GSM8K) =====")
    evaluator = GSM8KEvaluator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        use_4bit=True,
    )

    samples = evaluator.load_dataset(num_samples=num_samples, seed=0)
    results = evaluator.evaluate(samples, save_results=False)

    print("\nTask 1 summary (quick test):")
    print(f"  Total samples: {results['total']}")
    print(f"  Correct:       {results['correct']}")
    print(f"  Accuracy:      {results['accuracy']:.2%}")

    # free GPU memory if available
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_task2(num_samples: int = 3) -> None:
    print("\n===== Quick Test: Task 2 (ScienceQA) =====")
    evaluator = ScienceQAEvaluator(
        model_name="/root/autodl-tmp/models/Qwen2-VL-7B-Instruct",
        use_4bit=True,
    )

    samples = evaluator.load_dataset(num_samples=num_samples, seed=0)
    results = evaluator.evaluate(samples, save_results=False)

    print("\nTask 2 summary (quick test):")
    print(f"  Total samples: {results['total']}")
    print(f"  Correct:       {results['correct']}")
    print(f"  Accuracy:      {results['accuracy']:.2%}")

    # free GPU memory if available
    del evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    # run task1 quick test
    try:
        test_task1(num_samples=3)
    except Exception as e:
        print(f"Task 1 quick test failed: {e}")

    # run task2 quick test
    try:
        test_task2(num_samples=3)
    except Exception as e:
        print(f"Task 2 quick test failed: {e}")


if __name__ == "__main__":
    main()
