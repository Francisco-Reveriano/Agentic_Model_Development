import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from strands.models.anthropic import AnthropicModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Centralized configuration loaded from .env."""

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    model_id: str = "claude-opus-4-6"
    db_path: str = "Data/Raw/RCM_Controls.db"
    db_table: str = "my_table"
    output_dir: str = "Data/Output"
    pipeline_md_path: str = "docs/credit_risk_pd_lgd_ead_pipeline.md"
    max_tokens: int = 128_000
    log_level: str = "INFO"

    # Tournament configuration
    tournament_top_k: int = 5
    tournament_max_iterations: int = 5
    tournament_convergence_threshold: float = 0.002
    tournament_prune_threshold: float = 0.03
    tournament_cv_splits: int = 5
    tournament_random_search_iter: int = 50
    tournament_use_shap: bool = True
    tournament_feature_tiers: bool = True
    tournament_scoring_mode: str = "regulatory"

    model_config = {"env_file": str(PROJECT_ROOT / ".env"), "extra": "ignore"}

    # --- derived helpers ---

    @property
    def db_abs_path(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def output_abs_path(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def playbook_abs_path(self) -> Path:
        p = Path(self.pipeline_md_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def dictionary_abs_path(self) -> Path:
        return PROJECT_ROOT / "Data" / "RAW_FILES" / "LCDataDictionary.csv"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def create_anthropic_model(settings: Settings | None = None) -> AnthropicModel:
    """Factory matching the pattern from the reference Data_Agent.py."""
    s = settings or get_settings()
    return AnthropicModel(
        client_args={"api_key": s.anthropic_api_key},
        max_tokens=s.max_tokens,
        model_id=s.model_id,
        params={
            "temperature": 1,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
    )
