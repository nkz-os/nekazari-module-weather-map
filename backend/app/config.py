"""Weather Map configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    port: int = int(os.getenv("PORT", "8080"))
    
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "minio-service:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "nekazari-weather-map")
    minio_secure: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    
    elevation_service_url: str = os.getenv("ELEVATION_SERVICE_URL", "http://elevation-api-service:80")
    weather_api_url: str = os.getenv("WEATHER_API_URL", "http://weather-api-service:8000")
    orion_url: str = os.getenv("ORION_URL", "http://orion-ld-service:1026")
    
    cog_interval_days: int = int(os.getenv("COG_INTERVAL_DAYS", "5"))
    cog_retention_periods: int = int(os.getenv("COG_RETENTION_PERIODS", "3"))
    
    redis_url: str = os.getenv("REDIS_URL", "redis://redis-service:6379/0")
    dem_cache_ttl_days: int = int(os.getenv("DEM_CACHE_TTL_DAYS", "30"))
    
    auth_disabled: bool = os.getenv("AUTH_DISABLED", "false").lower() == "true"
    internal_service_secret: str = os.getenv("INTERNAL_SERVICE_SECRET", "")
    
    tile_size: int = 256
    tile_scale: float = 1.0
    
    metrics: list[str] = field(default_factory=lambda: [
        "temperature_avg", "temperature_min", "solar_radiation",
        "eto", "water_balance", "frost_risk", "soil_moisture",
    ])


settings = Settings()
