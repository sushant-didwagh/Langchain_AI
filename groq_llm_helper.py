import os
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from langchain_core.language_models.llms import LLM


class GroqLLM(LLM):
    model: str
    client: Groq

    def __init__(self, model: str = "llama-3.3-70b-versatile", client: Groq | None = None, **kwargs: Any):
        api_key = os.getenv("GROCK_API_KEY")
        if not api_key:
            raise ValueError("GROCK_API_KEY environment variable not set. Please add it to your .env file.")
        groq_client = client or Groq(api_key=api_key)
        super().__init__(model=model, client=groq_client, **kwargs)
        self.model = model
        self.client = groq_client

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}


class ChatGroq:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        client: Groq | None = None,
        temperature: float = 0.0,
        max_completion_tokens: int = 512,
    ):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        api_key = os.getenv("GROCK_API_KEY")
        if not api_key:
            raise ValueError("GROCK_API_KEY environment variable not set. Please add it to your .env file.")
        self.client = client or Groq(api_key=api_key)

    def invoke(self, messages: list[dict[str, str]] | str):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            messages=messages,
        )
        return SimpleNamespace(content=response.choices[0].message.content or "")


class GroqChatLLM(ChatGroq):
    pass


load_dotenv()


def get_groq_llm(model: str = "llama-3.3-70b-versatile") -> GroqLLM:
    return GroqLLM(model=model)


def get_groq_chat_llm(model: str = "llama-3.3-70b-versatile") -> GroqChatLLM:
    return GroqChatLLM(model=model)


def get_chat_groq_model(
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_completion_tokens: int = 512,
) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature, max_completion_tokens=max_completion_tokens)
