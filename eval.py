"""
Eval script (wrapper).

Bu dosya, LiveCodeBench değerlendirmesini nasıl çalıştırdığını göstermek için var.
Sonuçlar zaten results/livecodebench/ altında bulunuyor.
"""

def main():
    print("LiveCodeBench results are available under: results/livecodebench/")
    print("Main file: results/livecodebench/summary.json")
    print("\nIf you want to re-run evaluation, use the CodeGen repo command:")
    print("python livecodebench_eval.py --model_type deep_instruction --platform atcoder --difficulty easy")
    print("python livecodebench_eval.py --model_type diverse_instruction --platform atcoder --difficulty easy")

if __name__ == "__main__":
    main()
