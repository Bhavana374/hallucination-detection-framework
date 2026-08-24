"""
LLM Provider Abstraction for Generative AI Response Generation.

Supports Gemini, OpenAI, or local fallback response generation via env configuration.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger("llm_provider")


class LLMProvider(ABC):
    """Abstract base class for Generative AI response providers."""

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """Generate response text for prompt."""
        pass


class LocalFallbackProvider(LLMProvider):
    """Fallback response provider when external API keys are absent."""

    def generate_response(self, prompt: str) -> str:
        raise RuntimeError(
            "LLM API key not found in environment (.env). "
            "Please set GEMINI_API_KEY or OPENAI_API_KEY in .env to enable live LLM generation. "
            "You can still paste and analyze any AI-generated response directly."
        )


class GeminiProvider(LLMProvider):
    """Google Gemini API Provider wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment. Falling back to local provider.")
            self.fallback = LocalFallbackProvider()
        else:
            self.fallback = None

    def generate_response(self, prompt: str) -> str:
        if self.fallback:
            return self.fallback.generate_response(prompt)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content(prompt)
            return res.text.strip()
        except Exception as e:
            logger.error(f"Gemini API generation error: {e}. Using fallback.")
            return LocalFallbackProvider().generate_response(prompt)


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider wrapper."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment. Falling back to local provider.")
            self.fallback = LocalFallbackProvider()
        else:
            self.fallback = None

    def generate_response(self, prompt: str) -> str:
        if self.fallback:
            return self.fallback.generate_response(prompt)
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API generation error: {e}. Using fallback.")
            return LocalFallbackProvider().generate_response(prompt)


def get_llm_provider(provider_type: Optional[str] = None) -> LLMProvider:
    """Factory function to instantiate configured LLM provider."""
    provider_name = provider_type or os.getenv("LLM_PROVIDER", "auto").lower()

    if provider_name == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiProvider()
    elif provider_name == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    else:
        return LocalFallbackProvider()
