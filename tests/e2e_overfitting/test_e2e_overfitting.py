import os
import json
import shutil
import subprocess
import re
import pytest
import random
import numpy as np
import torch

from pathlib import Path


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch (CPU and CUDA) to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


CONFIG_PATH = "tests/e2e_overfitting/config.yaml"
OUTPUT_PATH = "tests/e2e_overfitting/output_dir"
GENERATE_PATH = "tests/e2e_overfitting/generate_outputs"

@pytest.fixture(scope="module", autouse=True)
def prepare_paths():
    shutil.rmtree(OUTPUT_PATH, ignore_errors=True)
    shutil.rmtree(GENERATE_PATH, ignore_errors=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(GENERATE_PATH, exist_ok=True)

def run_python_script(script_path, args):
    project_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        ["python", script_path] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(project_root),
        env={**os.environ,
             "CUDA_VISIBLE_DEVICES": "",
             "PYTORCH_ENABLE_MPS_FALLBACK": "1",
             "PYTHONPATH": str(project_root),
        },
    )
    print(f"\n--- Output of {script_path} ---\n{result.stdout}\n")
    assert result.returncode == 0, f"{script_path} failed"
    return result.stdout


def test_setup_runs_successfully():
    set_global_seed(42)
    _ = run_python_script(
        "multimodalhugs/multimodalhugs_cli/training_setup.py",
        ["--modality", "image2text", "--config_path", CONFIG_PATH, "--output_dir", OUTPUT_PATH],
    )


def test_model_converges_in_training():
    set_global_seed(42)
    run_python_script(
        "multimodalhugs/multimodalhugs_cli/train.py",
        [
            "--task", "translation",
            "--config_path", CONFIG_PATH,
            "--output_dir", OUTPUT_PATH,
            "--visualize_prediction_prob", "0",
            "--use_cpu",
            "--report_to", "none",
        ],
    )

    # === Read trainer_state.json from best checkpoint ===
    state_path = os.path.join(OUTPUT_PATH, "train", "trainer_state.json")
    if not os.path.exists(state_path):
        # fallback: try reading from best checkpoint path
        with open(os.path.join(OUTPUT_PATH, "train", "trainer_state.json")) as f:
            best_path = json.load(f)["best_model_checkpoint"]
            state_path = os.path.join(best_path, "trainer_state.json")

    with open(state_path, "r") as f:
        state = json.load(f)

    # Get the last eval_chrf reported in log_history
    eval_chrf_scores = [
        log["eval_chrf"] for log in state["log_history"]
        if "eval_chrf" in log
    ]
    assert eval_chrf_scores, "No eval_chrf found in trainer_state.json"
    best_eval_chrf = max(eval_chrf_scores)
    print(f"✅ best eval_chrf from trainer_state.json: {best_eval_chrf}")
    assert best_eval_chrf == 100.0, f"Expected eval_chrf of 100.0, got {best_eval_chrf}"


def test_generation_score_is_perfect():
    set_global_seed(42)
    # Find last checkpoint
    train_dir = os.path.join(OUTPUT_PATH, "train")
    search_path = train_dir if os.path.exists(train_dir) else OUTPUT_PATH

    checkpoints = [
        os.path.join(search_path, d)
        for d in os.listdir(search_path)
        if d.startswith("checkpoint-")
    ]

    assert checkpoints, "No checkpoint found"
    if any("checkpoint-best" in c for c in checkpoints):
        ckpt_path = [c for c in checkpoints if "checkpoint-best" in c][0]
    elif any("checkpoint-last" in c for c in checkpoints):
        ckpt_path = [c for c in checkpoints if "checkpoint-last" in c][0]
    else:
        ckpt_path = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

    # Run generation with two metrics to exercise the multi-metric code path
    _ = run_python_script(
        "multimodalhugs/multimodalhugs_cli/generate.py",
        [
            "--task", "translation",
            "--config_path", CONFIG_PATH,
            "--model_name_or_path", ckpt_path,
            "--metric_name", "sacrebleu,chrf",
            "--setup_path", f"{OUTPUT_PATH}/setup",
            "--generate_output_dir", GENERATE_PATH,
            "--do_predict", "true",
            "--use_cpu",
            "--max_length", "7",
            "--visualize_prediction_prob", "0",
            "--report_to", "none",
        ],
    )

    result_path = os.path.join(GENERATE_PATH, "predict_results.json")
    assert os.path.exists(result_path), "predict_results.json not found"

    with open(result_path, "r") as f:
        results = json.load(f)

    # Verify both metric keys are present (exercises zip loop and separate evaluate.load calls)
    assert "predict_chrf" in results, "predict_chrf not found in result file"
    assert "predict_sacrebleu" in results, "predict_sacrebleu not found in result file"
    print(f"✅ predict_chrf: {results['predict_chrf']}")
    print(f"✅ predict_sacrebleu: {results['predict_sacrebleu']}")
    assert results["predict_chrf"] == 100.0, f"Expected predict_chrf of 100.0, got {results['predict_chrf']}"
    assert results["predict_sacrebleu"] == 100.0, f"Expected predict_sacrebleu of 100.0, got {results['predict_sacrebleu']}"
