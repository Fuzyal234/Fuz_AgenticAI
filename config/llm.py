"""LLM configuration and client setup."""
from typing import Optional, List, Dict
from openai import OpenAI
from config.settings import settings


class LLMClient:
    """OpenAI/OpenRouter/Ollama LLM client wrapper."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, use_ollama: Optional[bool] = None):
        self.api_key = api_key or settings.openai_api_key
        self.use_ollama = use_ollama if use_ollama is not None else settings.use_ollama
        
        # Use Ollama if enabled (free local option)
        if self.use_ollama:
            self.model = model or settings.ollama_model
            # Ollama is OpenAI-compatible, point to /v1 endpoint
            ollama_url = settings.ollama_base_url.rstrip('/') + '/v1'
            self.client = OpenAI(
                api_key="ollama",  # Ollama doesn't require a real API key
                base_url=ollama_url
            )
            print(f"🦙 Using Ollama (FREE local LLM) with model: {self.model}")
        else:
            self.model = model or settings.openai_model
            # Detect if using OpenRouter (keys start with sk-or-v1-)
            if self.api_key and self.api_key.startswith("sk-or-v1-"):
                # Use OpenRouter endpoint
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
            else:
                # Use standard OpenAI endpoint
                self.client = OpenAI(api_key=self.api_key)
       
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate chat completion."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding


# Global LLM client instance
llm_client = LLMClient()

# LRM (Large Reasoning Model) client - uses specialized model for complex reasoning
# If Ollama is enabled, use it; otherwise use LRM_API_KEY or OPENAI_API_KEY
if settings.use_ollama:
    lrm_client = LLMClient(use_ollama=True, model=settings.ollama_model)
else:
    lrm_api_key = settings.lrm_api_key or settings.openai_api_key
    lrm_client = LLMClient(api_key=lrm_api_key, model=settings.lrm_model)

