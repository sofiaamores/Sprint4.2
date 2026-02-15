import os
from typing import Iterable, List, Dict, Optional, Tuple

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()


class AnthropicProvider:
    name = "Anthropic"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Falta ANTHROPIC_API_KEY en el entorno (.env).")

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        self.client = Anthropic(api_key=self.api_key)

    def _split_system_and_messages(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Anthropic separa `system` del array `messages`.
        Solo acepta roles: user / assistant.
        """
        system_parts = []
        out_messages = []

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if not content:
                continue

            if role == "system":
                system_parts.append(str(content))
            elif role in ("user", "assistant"):
                out_messages.append({"role": role, "content": str(content)})
            else:
                out_messages.append({"role": "user", "content": str(content)})

        system_text = "\n".join(system_parts).strip()
        return system_text, out_messages

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        model_name = model or self.model
        system_text, out_messages = self._split_system_and_messages(messages)

        resp = self.client.messages.create(
            model=model_name,
            system=system_text if system_text else None,
            messages=out_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Resp: lista de bloques; lo normal es texto en bloques type="text"
        parts = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

    def stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Iterable[str]:
        model_name = model or self.model
        system_text, out_messages = self._split_system_and_messages(messages)

        with self.client.messages.stream(
            model=model_name,
            system=system_text if system_text else None,
            messages=out_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            for event in stream:
                # Los deltas de texto suelen venir como content_block_delta con delta.text
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", None) if delta else None
                    if text:
                        yield text
