\# LoRA Fine-Tuning Project



This repository contains a LoRA-based fine-tuning and evaluation setup

for the Qwen2.5-Coder-1.5B-Instruct model.



\## Training

\- Training script: `train.py`

\- Two instruction settings:

&nbsp; - Deep Instruction

&nbsp; - Diverse Instruction

\- Training loss curves are provided under `logs/`.



\## Evaluation

\- Benchmark: LiveCodeBench (AtCoder, Easy)

\- Number of problems: 41

\- Evaluation script: `eval.py`

\- Results are available under `results/livecodebench/`



\## Results Summary

\- Base model Pass@1 ≈ 19.5%

\- Deep Instruction best Pass@1 ≈ 29%

\- Diverse Instruction best Pass@1 ≈ 34%



\## Notes

\- Model checkpoints are not included due to size constraints.

\- All benchmark outputs and logs are provided.



