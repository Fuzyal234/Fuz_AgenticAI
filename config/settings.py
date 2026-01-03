"""Configuration settings for FUZ_AgenticAI."""
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    
    # LLM Provider Selection
    use_ollama: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"  # Use Ollama for free local LLM
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Ollama server URL
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-coder")  # Default Ollama model
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4")
     
    # LRM (Large Reasoning Model) Configuration
    lrm_api_key: str = os.getenv("LRM_API_KEY", "")  # Separate API key for LRM
    lrm_model: str = os.getenv("LRM_MODEL", "gpt-4")  # Use gpt-4 or o1 for reasoning
    enable_lrm: bool = os.getenv("ENABLE_LRM", "true").lower() == "true"
    
    # Pinecone Configuration
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "fuz-agentic-ai")
    
    # GitHub Configuration
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_repo: str = os.getenv("GITHUB_REPO", "")
    github_base_branch: str = os.getenv("GITHUB_BASE_BRANCH", "main")
    
    # Agent Configuration
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "190"))
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

