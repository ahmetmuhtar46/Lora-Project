"""
Evaluation script for LoRA fine-tuned models.

This script documents how LiveCodeBench evaluation was performed
for the fine-tuned LoRA models.

Benchmark:
- LiveCodeBench (release_v5)
- Platform: AtCoder
- Difficulty: Easy
- Number of problems: 41

NOTE:
Actual benchmark outputs are already provided under:
results/livecodebench/
"""

import subprocess


def run_livecodebench(model_type, checkpoint_path=None):
    """
    Wrapper for LiveCodeBench evaluation command.
    """

    cmd = [
        "python",
        "livecodebench_eval.py",
        "--model_type", model_type,
        "--platform", "atcoder",
        "--difficulty", "easy"
    ]

    if checkpoint_path is not None:
        cmd.extend(["--checkpoint_path", checkpoint_path])

    print("Running command:")
    print(" ".join(cmd))

    # Uncomment the line below to actually run evaluation
    # subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # Base model
    run_livecodebench(model_type="base")

    # Deep Instruction LoRA checkpoints
    run_livecodebench(
        model_type="deep_instruction",
        checkpoint_path="models/deep_instruction/checkpoints/checkpoint-step-260-epoch-1"
    )

    run_livecodebench(
        model_type="deep_instruction",
        checkpoint_path="models/deep_instruction/checkpoints/checkpoint-step-300-epoch-1"
    )

    # Diverse Instruction LoRA checkpoints
    run_livecodebench(
        model_type="diverse_instruction",
        checkpoint_path="models/diverse_instruction/checkpoints/checkpoint-step-480-epoch-1"
    )

    run_livecodebench(
        model_type="diverse_instruction",
        checkpoint_path="models/diverse_instruction/checkpoints/checkpoint-step-620-epoch-1"
    )

    print("\nEvaluation completed.")
    print("See results in: results/livecodebench/")
