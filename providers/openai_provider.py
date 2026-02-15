import os
from typing import Iterable, List, Dict

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class OpenAIProvider:
    name = "AzureOpenAI"

    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01",
        )

        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        if not self.deployment_name:
            raise ValueError("Falta AZURE_OPENAI_DEPLOYMENT_NAME en el .env.")

    # --------------------------------
    # Construir prompt como texto plano
    # --------------------------------
    def _build_prompt(self, messages: List[Dict]) -> str:
        parts = []

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if not content:
                continue

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        return "\n".join(parts)

    # --------------------------------
    # No streaming
    # --------------------------------
    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> str:

        prompt = self._build_prompt(messages)

        response = self.client.responses.create(
            model=self.deployment_name,
            input=prompt,   # ⚠ SOLO STRING
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        return response.output_text

    # --------------------------------
    # Streaming
    # --------------------------------
    def stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
    ) -> Iterable[str]:

        prompt = self._build_prompt(messages)

        events = self.client.responses.create(
            model=self.deployment_name,
            input=prompt,   # ⚠ SOLO STRING
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            stream=True,
        )

        for event in events:
            if event.type == "response.output_text.delta":
                yield event.delta