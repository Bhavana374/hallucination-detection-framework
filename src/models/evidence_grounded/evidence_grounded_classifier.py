"""
Evidence-Grounded Hybrid Classifier for Hallucination Detection (Exp 5 & Exp 6).

Combines Transformer [CLS] representations (BERT or DeBERTa) with NLI posterior
probabilities (Entailment / Neutral / Contradiction) computed via NLICrossEncoder
into a feature fusion classification network.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from src.utils.logger import get_logger
from src.models.nli.nli_cross_encoder import NLICrossEncoder

logger = get_logger("evidence_grounded_classifier")

try:
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer, AutoModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch/Transformers unavailable. EvidenceGroundedClassifier using fallback mode.")


class FusionMLP(nn.Module if HAS_TORCH else object):
    """Multi-Layer Perceptron fusion head combining Transformer embedding and NLI probabilities."""

    def __init__(self, input_dim: int = 771, hidden_dim: int = 256, num_labels: int = 2, dropout: float = 0.2):
        if HAS_TORCH:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvidenceGroundedClassifier:
    """
    Evidence-Grounded Classifier for Exp 5 (BERT + NLI) and Exp 6 (DeBERTa + NLI).

    Architecture:
        Claim -> Transformer Encoder -> [CLS] embedding (d_model)
        (Evidence, Claim) -> NLI CrossEncoder -> [P_contra, P_ent, P_neut] (3d)
        Fusion Feature Vector = Concat([CLS], NLI_scores) (dim = d_model + 3)
        Fusion Feature Vector -> Fusion MLP Head -> Binary Classification (0=Factual, 1=Hallucinated)
    """

    def __init__(
        self,
        base_model_name: str = "google-bert/bert-base-uncased",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
        num_labels: int = 2,
        max_length: int = 256,
        freeze_base_encoder: bool = True,
        device: Optional[str] = None
    ):
        self.base_model_name = base_model_name
        self.nli_model_name = nli_model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.freeze_base_encoder = freeze_base_encoder
        self.is_fitted = False

        if HAS_TORCH:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                self.base_encoder = AutoModel.from_pretrained(base_model_name)
                self.base_encoder.to(self.device)

                if self.freeze_base_encoder:
                    for param in self.base_encoder.parameters():
                        param.requires_grad = False
                    self.base_encoder.eval()

                hidden_size = self.base_encoder.config.hidden_size
                fusion_input_dim = hidden_size + 3

                self.fusion_head = FusionMLP(input_dim=fusion_input_dim, hidden_dim=256, num_labels=num_labels)
                self.fusion_head.to(self.device)

                self.nli_scorer = NLICrossEncoder(model_name=nli_model_name, device=self.device)
                logger.info(
                    f"Initialized EvidenceGroundedClassifier: base='{base_model_name}', "
                    f"nli='{nli_model_name}', fusion_dim={fusion_input_dim}, freeze_base={freeze_base_encoder}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize EvidenceGroundedClassifier models: {e}")
                self.tokenizer = None
                self.base_encoder = None
                self.fusion_head = None
                self.nli_scorer = None
        else:
            self.device = "cpu"
            self.tokenizer = None
            self.base_encoder = None
            self.fusion_head = None
            self.nli_scorer = None

    def _extract_fusion_features(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16
    ) -> torch.Tensor:
        """Extract concatenated ([CLS] embedding + NLI probabilities) feature tensor."""
        if not HAS_TORCH or self.tokenizer is None or self.base_encoder is None:
            raise RuntimeError("PyTorch/Transformers unavailable for feature extraction.")

        all_features = []
        n_samples = len(claims)

        for i in range(0, n_samples, batch_size):
            batch_claims = claims[i : i + batch_size]
            batch_evidences = evidences[i : i + batch_size]

            # 1. Base Encoder [CLS] representation
            inputs = self.tokenizer(
                batch_claims,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if self.freeze_base_encoder:
                with torch.no_grad():
                    outputs = self.base_encoder(**inputs)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            else:
                outputs = self.base_encoder(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # 2. NLI Cross-Encoder probabilities
            nli_pairs = list(zip(batch_evidences, batch_claims))
            nli_scores = self.nli_scorer.predict(nli_pairs)  # np.ndarray [batch_size, 3]
            nli_tensor = torch.tensor(nli_scores, dtype=torch.float32, device=self.device)

            # 3. Concatenate
            fused_batch = torch.cat([cls_embeddings, nli_tensor], dim=1)
            all_features.append(fused_batch)

        return torch.cat(all_features, dim=0)

    def fit(
        self,
        train_claims: List[str],
        train_evidences: List[str],
        train_labels: List[int],
        val_claims: Optional[List[str]] = None,
        val_evidences: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        patience: int = 3
    ) -> Dict[str, Any]:
        """
        Fit the fusion classification head on (claim, evidence) pairs and binary labels.
        """
        if not HAS_TORCH or self.fusion_head is None:
            logger.warning("PyTorch not installed. Skipping training.")
            return {}

        start_time = time.time()
        logger.info(f"Extracting fusion features for {len(train_claims)} training samples...")

        train_features = self._extract_fusion_features(train_claims, train_evidences, batch_size=batch_size)
        y_train_tensor = torch.tensor(train_labels, dtype=torch.long, device=self.device)

        if val_claims is not None and val_evidences is not None and val_labels is not None:
            val_features = self._extract_fusion_features(val_claims, val_evidences, batch_size=batch_size)
            y_val_tensor = torch.tensor(val_labels, dtype=torch.long, device=self.device)
        else:
            val_features = None
            y_val_tensor = None

        dataset = torch.utils.data.TensorDataset(train_features, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.fusion_head.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.fusion_head.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                logits = self.fusion_head(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)

            avg_train_loss = total_loss / len(train_claims)

            if val_features is not None and y_val_tensor is not None:
                self.fusion_head.eval()
                with torch.no_grad():
                    val_logits = self.fusion_head(val_features)
                    val_loss = loss_fn(val_logits, y_val_tensor).item()
                self.fusion_head.train()

                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.fusion_head.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}")

        if best_state is not None:
            self.fusion_head.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        self.fusion_head.eval()
        self.is_fitted = True
        train_time = time.time() - start_time
        logger.info(f"EvidenceGroundedClassifier training completed in {train_time:.2f}s")
        return {"training_time_sec": train_time}

    def predict_proba(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16
    ) -> np.ndarray:
        """Predict binary label probabilities for claim-evidence pairs."""
        if not claims:
            return np.zeros((0, self.num_labels), dtype=np.float32)

        if HAS_TORCH and self.fusion_head is not None:
            self.fusion_head.eval()
            features = self._extract_fusion_features(claims, evidences, batch_size=batch_size)
            with torch.no_grad():
                logits = self.fusion_head(features)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs.astype(np.float32)
        else:
            # Fallback uniform probs
            return np.full((len(claims), self.num_labels), 0.5, dtype=np.float32)

    def predict(
        self,
        claims: List[str],
        evidences: List[str],
        batch_size: int = 16
    ) -> List[int]:
        """Predict binary labels for claim-evidence pairs."""
        probs = self.predict_proba(claims, evidences, batch_size=batch_size)
        return list(np.argmax(probs, axis=1))

    def save(self, output_dir: Union[str, Path]) -> None:
        """Save model checkpoint."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if HAS_TORCH and self.fusion_head is not None:
            torch.save(self.fusion_head.state_dict(), output_dir / "fusion_head.pt")
            logger.info(f"Saved EvidenceGroundedClassifier checkpoint to {output_dir}")

    def load(self, model_dir: Union[str, Path]) -> None:
        """Load model checkpoint."""
        model_dir = Path(model_dir)
        ckpt = model_dir / "fusion_head.pt"
        if HAS_TORCH and os.path.exists(ckpt):
            self.fusion_head.load_state_dict(torch.load(ckpt, map_location=self.device))
            self.fusion_head.eval()
            self.is_fitted = True
            logger.info(f"Loaded EvidenceGroundedClassifier checkpoint from {model_dir}")
