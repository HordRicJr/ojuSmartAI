from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Paramètres de configuration globaux du microservice IA."""

    app_name: str = "OjuSmart Stateless AI Engine"
    app_version: str = "1.0.0"
    debug: bool = False
    embedding_dimension: int = 512
    image_target_size: int = 224
    signature_similarity_threshold: float = 0.92
    scene_detection_threshold: float = 0.7
    scene_max_objects: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
