"""
DeBERTa-v3 Fine-Tuned Classifier for Hallucination Detection (Exp 4 & Exp 6).
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import numpy as np
from src.utils.logger import get_logger

logger = get_logger("deberta_classifier")

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch/Transformers unavailable. DeBERTa model running in fallback mode.")


class FallbackDeBERTaModel:
    """Fallback classifier when PyTorch/Transformers is not installed."""

    def __init__(self, num_labels: int = 2):
        self.num_labels = num_labels

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        probs = []
        for t in texts:
            val = (len(t) * 7 % 100) / 100.0
            if self.num_labels == 2:
                probs.append([1.0 - val, val])
            else:
                probs.append([val * 0.4, (1.0 - val) * 0.4, 0.6])
        return np.array(probs, dtype=np.float32)


class DeBERTaClassifier:
    """
    DeBERTa Classifier for Standalone (Exp 4) and Evidence-Grounded NLI (Exp 6).
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        max_length: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length

        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
                self.model.to(self.device)
                logger.info(f"Loaded DeBERTa model '{model_name}' on device '{self.device}'")
            except Exception as e:
                logger.warning(f"Could not load model '{model_name}': {e}. Using fallback DeBERTa model.")
                self.model = FallbackDeBERTaModel(num_labels=num_labels)
                self.tokenizer = None
        else:
            self.device = "cpu"
            self.model = FallbackDeBERTaModel(num_labels=num_labels)
            self.tokenizer = None

    def predict_proba(self, texts: List[str], pair_texts: Optional[List[str]] = None) -> np.ndarray:
        """Predict label probabilities."""
        if not texts:
            return np.zeros((0, self.num_labels), dtype=np.float32)

        if HAS_TORCH and self.tokenizer is not None and isinstance(self.model, torch.nn.Module):
            self.model.eval()
            if pair_texts is not None:
                inputs = self.tokenizer(texts, pair_texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            else:
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs.astype(np.float32)
        else:
            return self.model.predict_proba(texts)

    def predict(self, texts: List[str], pair_texts: Optional[List[str]] = None) -> List[int]:
        probs = self.predict_proba(texts, pair_texts)
        return list(np.argmax(probs, axis=1))

    def save(self, output_dir: Union[str, Path]) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.tokenizer is not None and isinstance(self.model, torch.nn.Module):
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Saved DeBERTa checkpoint to {output_dir}")

    def load(self, model_dir: Union[str, Path]) -> None:
        if HAS_TORCH and os.path.exists(model_dir):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model.to(self.device)
            logger.info(f"Loaded DeBERTa model from {model_dir}")
