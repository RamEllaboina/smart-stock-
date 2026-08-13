import os

class Settings:
    PROJECT_NAME: str = "Smart Stock Secure API"
    SECRET_KEY: str = os.getenv("JWT_SECRET", "change-me-in-production")
    ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./smartstock.db")
    MAX_PAGE_SIZE: int = int(os.getenv("MAX_PAGE_SIZE", 100))
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    # Rate Limiting Configurations (simulated)
    RATE_LIMIT_LOGIN: str = "10/minute"

settings = Settings()
