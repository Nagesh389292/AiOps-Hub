import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _inject_streamlit_secrets() -> None:
    """Push Streamlit Cloud secrets into os.environ so pydantic-settings can read them."""
    try:
        import streamlit as st
        secrets = dict(st.secrets)
        for key, value in secrets.items():
            if key not in os.environ:
                os.environ[key] = str(value)
    except Exception:
        pass  # Not running under Streamlit, or secrets not configured yet


_inject_streamlit_secrets()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "AIOps Hub"
    runtime_mode: str = Field(default="real", alias="AIOPS_RUNTIME_MODE")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")

    database_url: str = Field(default="sqlite:///aiops_hub.db", alias="DATABASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    recommendation_weight_accuracy: float = Field(default=0.40, alias="RECOMMENDATION_WEIGHT_ACCURACY")
    recommendation_weight_validation: float = Field(default=0.30, alias="RECOMMENDATION_WEIGHT_VALIDATION")
    recommendation_weight_latency: float = Field(default=0.15, alias="RECOMMENDATION_WEIGHT_LATENCY")
    recommendation_weight_cost: float = Field(default=0.15, alias="RECOMMENDATION_WEIGHT_COST")

    min_required_accuracy: float = Field(default=85.0, alias="MIN_REQUIRED_ACCURACY")
    min_required_validation_success: float = Field(default=90.0, alias="MIN_REQUIRED_VALIDATION_SUCCESS")
    max_allowed_average_latency_ms: float = Field(default=2500.0, alias="MAX_ALLOWED_AVERAGE_LATENCY_MS")

    provider_max_retries: int = Field(default=2, alias="PROVIDER_MAX_RETRIES")
    provider_retry_backoff_seconds: float = Field(default=0.5, alias="PROVIDER_RETRY_BACKOFF_SECONDS")

    dataset_version: str = Field(default="v1", alias="DATASET_VERSION")
    prompt_template_version: str = Field(default="v1", alias="PROMPT_TEMPLATE_VERSION")
    evaluation_rubric_version: str = Field(default="v1", alias="EVALUATION_RUBRIC_VERSION")

    base_dir: Path = Path(__file__).resolve().parent


settings = Settings()
