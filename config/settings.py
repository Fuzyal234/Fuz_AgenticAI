"""Configuration settings for FUZ_AgenticAI."""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Pinecone Configuration
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "fuz-agentic-ai")
    
    # GitHub Configuration
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_repo: str = os.getenv("GITHUB_REPO", "")
    github_base_branch: str = os.getenv("GITHUB_BASE_BRANCH", "main")
    
    # Agent Configuration
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    enable_auto_fix: bool = os.getenv("ENABLE_AUTO_FIX", "true").lower() == "true"
    
    # Shell Command Allow-list
    allowed_commands: list = [
        "python", "pytest", "npm", "yarn", "pip", "git",
        "make", "docker", "black", "flake8", "mypy"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

