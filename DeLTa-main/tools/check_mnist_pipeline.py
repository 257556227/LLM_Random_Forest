import argparse
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Check MNIST pipeline artifacts and metrics")
    p.add_argument("--project_root", type=str, default=".")
    p.add_argument("--min_fused_acc", type=float, default=0.90)
    p.add_argument("--expect_answers", type=int, default=10)
    return p.parse_args()


def read_fused_acc(log_path: Path) -> float:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"Fused Accuracy MEAN:\s*([0-9.]+)", text)
    if not m:
        raise RuntimeError(f"Cannot find Fused Accuracy in {log_path}")
    return float(m.group(1))


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()

    required = [
        root / "llm" / "prompts" / "mnist" / "mnist_randfull_md8_ml20_tree3.txt",
        root / "model" / "llm_rule" / "mnist.py",
        root / "results" / "mnist" / "mnist_md8_ml20_tree3.npy",
        root / "results" / "mnist" / "e_RF_md8_ml20_tree3_mnist_cart_9.log",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Missing required artifacts:\n" + "\n".join(missing))

    answers_dir = root / "llm" / "answers" / "mnist"
    answer_files = sorted(answers_dir.glob("mnist_randfull_md8_ml20_tree3_*.txt"))
    if len(answer_files) < args.expect_answers:
        raise RuntimeError(
            f"Expected at least {args.expect_answers} answers, got {len(answer_files)}"
        )

    fused_acc = read_fused_acc(root / "results" / "mnist" / "e_RF_md8_ml20_tree3_mnist_cart_9.log")
    if fused_acc < args.min_fused_acc:
        raise RuntimeError(
            f"Fused accuracy {fused_acc:.4f} < threshold {args.min_fused_acc:.4f}"
        )

    print("MNIST pipeline check passed")
    print(f"answers={len(answer_files)}")
    print(f"fused_acc={fused_acc:.4f}")


if __name__ == "__main__":
    main()
