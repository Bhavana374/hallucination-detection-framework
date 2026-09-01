"""
Model Setup Verification Script for Hallucination Detection Framework.

Verifies that all three models required for Experiments 3–6 can be
downloaded, loaded, and run inference successfully:

    1. google-bert/bert-base-uncased          (Exp 3 & 5)
    2. microsoft/deberta-v3-base              (Exp 4 & 6)
    3. cross-encoder/nli-deberta-v3-small     (Exp 5 & 6 — NLI evidence scoring)

Usage:
    python scripts/verify_model_setup.py
"""

import sys
import os
import io
import time
import json
import traceback

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===================================================================
# Helper utilities
# ===================================================================

def _section(title: str) -> None:
    width = 68
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _ok(msg: str) -> None:
    print(f"  [✓] {msg}")


def _fail(msg: str) -> None:
    print(f"  [✗] {msg}")


def _info(msg: str) -> None:
    print(f"  [i] {msg}")


# ===================================================================
# Step 0 — Hardware detection
# ===================================================================

def detect_hardware() -> dict:
    """Detect available compute device and return metadata dict."""
    _section("STEP 0 — Hardware Detection")

    hw = {
        "torch_installed": False,
        "torch_version": None,
        "device": "cpu",
        "device_name": "CPU",
        "gpu_memory_gb": None,
    }

    try:
        import torch
        hw["torch_installed"] = True
        hw["torch_version"] = torch.__version__
        _ok(f"PyTorch installed — version {torch.__version__}")

        if torch.cuda.is_available():
            hw["device"] = "cuda"
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            hw["device_name"] = props.name
            hw["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            _ok(f"Device: CUDA")
            _ok(f"GPU: {props.name}")
            _ok(f"GPU Memory: {hw['gpu_memory_gb']} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            hw["device"] = "mps"
            hw["device_name"] = "Apple MPS"
            _ok(f"Device: MPS (Apple Silicon)")
        else:
            hw["device"] = "cpu"
            hw["device_name"] = "CPU"
            _ok(f"Device: CPU")

    except ImportError:
        _fail("PyTorch is NOT installed.")

    # Check transformers
    try:
        import transformers
        _ok(f"Transformers installed — version {transformers.__version__}")
        hw["transformers_version"] = transformers.__version__
    except ImportError:
        _fail("Transformers is NOT installed.")
        hw["transformers_version"] = None

    # Check sentence-transformers
    try:
        import sentence_transformers
        _ok(f"Sentence-Transformers installed — version {sentence_transformers.__version__}")
        hw["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        _fail("Sentence-Transformers is NOT installed.")
        hw["sentence_transformers_version"] = None

    # Hugging Face cache location
    from huggingface_hub import constants as hf_constants
    cache_dir = getattr(hf_constants, "HF_HUB_CACHE", None) or os.environ.get(
        "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    hw["hf_cache_dir"] = str(cache_dir)
    _info(f"HuggingFace cache: {cache_dir}")

    return hw


# ===================================================================
# Step 1 — BERT verification  (google-bert/bert-base-uncased)
# ===================================================================

def verify_bert(device: str) -> dict:
    """Download, load, and run dummy inference with BERT."""
    _section("STEP 1 — BERT (google-bert/bert-base-uncased)")

    MODEL_NAME = "google-bert/bert-base-uncased"
    result = {"model_id": MODEL_NAME, "status": "FAILED", "device": device}

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # --- Load tokenizer ---
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _ok(f"Tokenizer loaded ({time.time() - t0:.1f}s)")

        # --- Load model ---
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        model.to(device)
        model.eval()
        _ok(f"Model loaded on '{device}' ({time.time() - t0:.1f}s)")
        _info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # --- Dummy inference ---
        question = "Who wrote Hamlet?"
        answer = "William Shakespeare wrote Hamlet."
        text = f"{question} [SEP] {answer}"

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        logits_list = logits.cpu().squeeze().tolist()
        probs_list = probs.cpu().squeeze().tolist()

        _ok(f"Inference successful")
        _info(f"Input:  \"{text}\"")
        _info(f"Logits: {[round(x, 4) for x in logits_list]}")
        _info(f"Probs:  {[round(x, 4) for x in probs_list]}")
        _info(f"Labels: [0 = Factual, 1 = Hallucinated]")

        result["status"] = "OK"
        result["logits"] = logits_list
        result["probs"] = probs_list

    except Exception as e:
        _fail(f"BERT verification failed: {e}")
        traceback.print_exc()
        result["error"] = str(e)

    return result


# ===================================================================
# Step 2 — DeBERTa verification  (microsoft/deberta-v3-base)
# ===================================================================

def verify_deberta(device: str) -> dict:
    """Download, load, and run dummy inference with DeBERTa-v3-base."""
    _section("STEP 2 — DeBERTa (microsoft/deberta-v3-base)")

    MODEL_NAME = "microsoft/deberta-v3-base"
    result = {"model_id": MODEL_NAME, "status": "FAILED", "device": device}

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # --- Load tokenizer ---
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _ok(f"Tokenizer loaded ({time.time() - t0:.1f}s)")

        # --- Load model ---
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        )
        model.to(device)
        model.eval()
        _ok(f"Model loaded on '{device}' ({time.time() - t0:.1f}s)")
        _info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # --- Dummy inference ---
        question = "Who wrote Hamlet?"
        answer = "William Shakespeare wrote Hamlet."
        text = f"{question} [SEP] {answer}"

        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        logits_list = logits.cpu().squeeze().tolist()
        probs_list = probs.cpu().squeeze().tolist()

        _ok(f"Inference successful")
        _info(f"Input:  \"{text}\"")
        _info(f"Logits: {[round(x, 4) for x in logits_list]}")
        _info(f"Probs:  {[round(x, 4) for x in probs_list]}")
        _info(f"Labels: [0 = Factual, 1 = Hallucinated]")

        result["status"] = "OK"
        result["logits"] = logits_list
        result["probs"] = probs_list

    except Exception as e:
        _fail(f"DeBERTa verification failed: {e}")
        traceback.print_exc()
        result["error"] = str(e)

    return result


# ===================================================================
# Step 3 — NLI Cross-Encoder  (cross-encoder/nli-deberta-v3-small)
# ===================================================================

def verify_nli(device: str) -> dict:
    """Download, load, and run dummy NLI inference."""
    _section("STEP 3 — NLI Cross-Encoder (cross-encoder/nli-deberta-v3-small)")

    MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
    result = {"model_id": MODEL_NAME, "status": "FAILED", "device": device}

    try:
        from sentence_transformers import CrossEncoder

        # --- Load model ---
        t0 = time.time()
        nli_model = CrossEncoder(MODEL_NAME)
        _ok(f"NLI model loaded ({time.time() - t0:.1f}s)")

        # --- Resolve label mapping from config ---
        label_mapping = {}
        try:
            config = nli_model.model.config
            if hasattr(config, "id2label") and config.id2label:
                label_mapping = {
                    int(k): v for k, v in config.id2label.items()
                }
                _ok(f"Label mapping from config: {label_mapping}")
        except Exception:
            pass

        if not label_mapping:
            label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
            _info(f"Using default label mapping: {label_mapping}")

        # --- Dummy NLI inference ---
        premise = "Berlin is the capital of Germany."
        hypothesis = "Paris is the capital of Germany."

        scores = nli_model.predict(
            [(premise, hypothesis)], apply_softmax=True
        )

        scores_arr = scores[0] if hasattr(scores[0], "__len__") else scores
        score_dict = {
            label_mapping[i]: round(float(s), 4) for i, s in enumerate(scores_arr)
        }

        _ok(f"NLI inference successful")
        _info(f"Premise:    \"{premise}\"")
        _info(f"Hypothesis: \"{hypothesis}\"")
        _info(f"Scores:     {score_dict}")

        result["status"] = "OK"
        result["label_mapping"] = {str(k): v for k, v in label_mapping.items()}
        result["scores"] = score_dict

    except Exception as e:
        _fail(f"NLI verification failed: {e}")
        traceback.print_exc()
        result["error"] = str(e)

    return result


# ===================================================================
# Final report
# ===================================================================

def print_summary(hw: dict, bert: dict, deberta: dict, nli: dict) -> None:
    _section("SETUP VERIFICATION SUMMARY")

    rows = [
        ("BERT",    bert["model_id"],    bert["status"],    bert.get("device", "?")),
        ("DeBERTa", deberta["model_id"], deberta["status"], deberta.get("device", "?")),
        ("NLI",     nli["model_id"],     nli["status"],     nli.get("device", "?")),
    ]

    for name, mid, status, dev in rows:
        icon = "✓" if status == "OK" else "✗"
        print(f"  [{icon}] {name}")
        print(f"      Model ID : {mid}")
        print(f"      Status   : {status}")
        print(f"      Device   : {dev}")
        print()

    print(f"  HuggingFace cache : {hw.get('hf_cache_dir', 'unknown')}")
    print(f"  PyTorch version   : {hw.get('torch_version', 'N/A')}")
    print(f"  Transformers ver. : {hw.get('transformers_version', 'N/A')}")
    print(f"  Sentence-Trans.   : {hw.get('sentence_transformers_version', 'N/A')}")
    print()

    # Checklist
    checks = {
        "Transformers installed":       hw.get("transformers_version") is not None,
        "PyTorch installed":            hw.get("torch_installed", False),
        "BERT downloaded successfully": bert["status"] == "OK",
        "BERT tokenizer loaded":        bert["status"] == "OK",
        "BERT inference works":         bert["status"] == "OK",
        "DeBERTa downloaded successfully": deberta["status"] == "OK",
        "DeBERTa tokenizer loaded":     deberta["status"] == "OK",
        "DeBERTa inference works":      deberta["status"] == "OK",
        "NLI model downloaded successfully": nli["status"] == "OK",
        "NLI inference works":          nli["status"] == "OK",
    }

    print("  Checklist:")
    for label, ok in checks.items():
        icon = "✓" if ok else "✗"
        print(f"    [{icon}] {label}")

    all_ok = all(checks.values())
    print()
    if all_ok:
        print("  ✅ All models verified successfully. Ready for Experiments 3–6.")
    else:
        print("  ⚠️  Some checks failed. See errors above.")


def save_report(hw: dict, bert: dict, deberta: dict, nli: dict) -> Path:
    """Persist a JSON report under results/."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw,
        "bert": bert,
        "deberta": deberta,
        "nli": nli,
    }

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "model_setup_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    _info(f"Setup report saved to {report_path}")
    return report_path


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    print()
    print("+" + "=" * 68 + "+")
    print("|   Hallucination Detection Framework - Model Setup Verification   |")
    print("+" + "=" * 68 + "+")

    hw = detect_hardware()
    device = hw["device"]

    bert_result = verify_bert(device)
    deberta_result = verify_deberta(device)
    nli_result = verify_nli(device)

    print_summary(hw, bert_result, deberta_result, nli_result)
    save_report(hw, bert_result, deberta_result, nli_result)

    print()
    print("  Done. No training has been started.")
    print("  To repeat this check, run:")
    print()
    print("    python scripts/verify_model_setup.py")
    print()


if __name__ == "__main__":
    main()
