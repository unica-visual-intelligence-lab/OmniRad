import subprocess
import sys
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"

NUM_ITERS = 10
USE_LORA = [False]
MODELS = [
    "hf_hub:Snarcy/tbd_s",
    "hf_hub:Snarcy/tbd_b",
]

DATASETS = [
    "breastmnist",
    "pneumoniamnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]

OUTPUT_PATH_ROOT = "path/to/finetuning_outputs"


def run_for_model(
    model_name,
    dataset_name,
    train_path,
    val_path,
    test_path,
    output_path,
    use_lora,
) -> int:
    """Call train.py once for a given model name."""

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model", model_name,
        "--dataset_name", dataset_name,
        "--dataset_path_train", train_path,
        "--dataset_path_val", val_path,
        "--dataset_path_test", test_path,
        "--output_path", output_path,
    ]

    if use_lora:
        cmd.append("--use_lora")

    print("\n========================================")
    print(f"Running train.py for model: {model_name}")
    print("Command:", " ".join(cmd))
    print("========================================\n")

    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    for iteration in range(NUM_ITERS):
        for use_lora in USE_LORA:
            if use_lora:
                output_path_root_use = os.path.join(OUTPUT_PATH_ROOT, "lora")
            else:
                output_path_root_use = os.path.join(OUTPUT_PATH_ROOT, "full_tuning")
            for model_name in MODELS:
                for dataset_name in DATASETS:
                    train_path = f"path/to/data_split/{dataset_name}_224/train"
                    val_path = f"path/to/data_split/{dataset_name}_224/validation"
                    test_path = f"path/to/data_split/{dataset_name}_224/test"

                    output_path = os.path.join(
                        output_path_root_use, f"iteration_{iteration}"
                    )

                    ret = run_for_model(
                        model_name,
                        dataset_name,
                        train_path,
                        val_path,
                        test_path,
                        output_path,
                        use_lora,
                    )

                    if ret != 0:
                        print(
                            f"train.py failed for model {model_name} with code {ret}"
                        )


if __name__ == "__main__":
    main()
