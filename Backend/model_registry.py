"""Model registry for tracking trained models across pipeline runs.

Provides persistent JSON-based storage of model metadata, metrics,
hyperparameters, and champion designation. Supports querying for
champion models and listing/filtering registered models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelRegistry:
    """JSON-backed registry for trained credit risk models.

    Each entry contains:
      - model_id: unique identifier (model_type_algorithm_timestamp)
      - model_type: PD, LGD, or EAD
      - algorithm: e.g. xgboost, lightgbm, logistic_regression
      - metrics: dict of performance metrics
      - hyperparameters: dict of model hyperparameters
      - model_path: path to the serialized model artifact
      - champion: bool flag indicating current champion model
      - dataset_hash: hash of the training dataset for reproducibility
      - feature_count: number of features used
      - training_time_s: training duration in seconds
      - registered_at: ISO-8601 timestamp
    """

    def __init__(self, output_dir: Path):
        """Initialize the model registry.

        Args:
            output_dir: Directory where model_registry.json is stored.
        """
        self.output_dir = Path(output_dir)
        self.registry_path = self.output_dir / "model_registry.json"
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        """Create the registry file if it does not exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]")

    def _load(self) -> list[dict[str, Any]]:
        """Load the registry from disk.

        Returns:
            List of model entry dicts.
        """
        try:
            text = self.registry_path.read_text().strip()
            if not text:
                return []
            entries = json.loads(text)
            if not isinstance(entries, list):
                return []
            return entries
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        """Persist the registry to disk.

        Args:
            entries: List of model entry dicts to save.
        """
        self.registry_path.write_text(
            json.dumps(entries, indent=2, default=str)
        )

    def register(
        self,
        model_type: str,
        algorithm: str,
        metrics: dict[str, Any],
        hyperparameters: dict[str, Any],
        model_path: str,
        champion: bool,
        dataset_hash: str = "",
        feature_count: int = 0,
        training_time_s: float | None = None,
    ) -> str:
        """Register a trained model in the registry.

        If champion=True, any existing champion for the same model_type is
        automatically demoted (champion set to False).

        Args:
            model_type: Model type (PD, LGD, EAD).
            algorithm: Algorithm name (e.g., xgboost, lightgbm).
            metrics: Performance metrics dict (e.g., {"auc_roc": 0.85, "gini": 0.70}).
            hyperparameters: Model hyperparameters dict.
            model_path: File path to the serialized model artifact.
            champion: Whether this is the champion model for its type.
            dataset_hash: Hash of the training dataset for reproducibility tracking.
            feature_count: Number of features used in the model.
            training_time_s: Training duration in seconds.

        Returns:
            The generated model_id string.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_id = f"{model_type.upper()}_{algorithm}_{timestamp}"

        entries = self._load()

        # Demote existing champion if registering a new one
        if champion:
            for entry in entries:
                if (
                    entry.get("model_type", "").upper() == model_type.upper()
                    and entry.get("champion") is True
                ):
                    entry["champion"] = False

        new_entry: dict[str, Any] = {
            "model_id": model_id,
            "model_type": model_type.upper(),
            "algorithm": algorithm,
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "model_path": str(model_path),
            "champion": champion,
            "dataset_hash": dataset_hash,
            "feature_count": feature_count,
            "training_time_s": training_time_s,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

        entries.append(new_entry)
        self._save(entries)

        return model_id

    def get_champion(self, model_type: str) -> dict[str, Any] | None:
        """Find the current champion model for a given model type.

        Args:
            model_type: Model type to search for (PD, LGD, EAD).

        Returns:
            The champion model entry dict, or None if no champion is registered.
        """
        entries = self._load()
        for entry in reversed(entries):  # most recent first
            if (
                entry.get("model_type", "").upper() == model_type.upper()
                and entry.get("champion") is True
            ):
                return entry
        return None

    def list_models(self, model_type: str | None = None) -> list[dict[str, Any]]:
        """List registered models, optionally filtered by model type.

        Args:
            model_type: If provided, filter to only this model type (PD, LGD, EAD).
                        If None, return all registered models.

        Returns:
            List of model entry dicts, sorted by registration time (newest first).
        """
        entries = self._load()

        if model_type is not None:
            entries = [
                e for e in entries
                if e.get("model_type", "").upper() == model_type.upper()
            ]

        # Sort by registered_at descending (newest first)
        entries.sort(key=lambda e: e.get("registered_at", ""), reverse=True)
        return entries

    def promote_champion(self, model_id: str) -> bool:
        """Promote a specific model to champion status.

        Demotes any existing champion of the same model type.

        Args:
            model_id: The model_id to promote.

        Returns:
            True if the model was found and promoted, False otherwise.
        """
        entries = self._load()
        target_entry = None

        for entry in entries:
            if entry.get("model_id") == model_id:
                target_entry = entry
                break

        if target_entry is None:
            return False

        target_type = target_entry.get("model_type", "").upper()

        # Demote existing champion
        for entry in entries:
            if (
                entry.get("model_type", "").upper() == target_type
                and entry.get("champion") is True
            ):
                entry["champion"] = False

        # Promote target
        target_entry["champion"] = True
        self._save(entries)
        return True

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Retrieve a specific model entry by its model_id.

        Args:
            model_id: The unique model identifier.

        Returns:
            The model entry dict, or None if not found.
        """
        entries = self._load()
        for entry in entries:
            if entry.get("model_id") == model_id:
                return entry
        return None

    def summary(self) -> dict[str, Any]:
        """Generate a summary of the registry contents.

        Returns:
            Dict with total count, per-type counts, and champion info.
        """
        entries = self._load()
        type_counts: dict[str, int] = {}
        champions: dict[str, str] = {}

        for entry in entries:
            mt = entry.get("model_type", "UNKNOWN")
            type_counts[mt] = type_counts.get(mt, 0) + 1
            if entry.get("champion"):
                champions[mt] = entry.get("model_id", "")

        return {
            "total_models": len(entries),
            "by_type": type_counts,
            "champions": champions,
            "registry_path": str(self.registry_path),
        }
