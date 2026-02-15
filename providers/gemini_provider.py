import os
from typing import Iterable, List, Dict, Tuple, Optional

from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig

load_dotenv()


class GeminiProvider:
    name = "Gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Falta GEMINI_API_KEY en el entorno (.env).")

        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=self.api_key)

    def _split_system_and_contents(self, messages: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Gemini no usa role 'system' en el historial.
        - Extraemos system -> system_instruction
        - user -> role 'user'
        - assistant -> role 'model'
        """
        system_parts = []
        contents = []

        for m in messages:
            role = m.get("role")
            content = m.get("content", "")

            if not content:
                continue

            if role == "system":
                system_parts.append(str(content))
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": str(content)}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": str(content)}]})
            else:
                # Si llega algo raro, lo tratamos como user para no romper
                contents.append({"role": "user", "parts": [{"text": str(content)}]})

        system_instruction = "\n".join(system_parts).strip()
        return system_instruction, contents

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        model_name = model or self.model
        system_instruction, contents = self._split_system_and_contents(messages)

        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None,
        )

        resp = self.client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        # `text` suele traer la respuesta agregada
        return getattr(resp, "text", "") or ""

    def stream(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> Iterable[str]:
        model_name = model or self.model
        system_instruction, contents = self._split_system_and_contents(messages)

        config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction if system_instruction else None,
        )

        stream = self.client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=config,
        )

        for chunk in stream:
            # Cada chunk suele traer `text` parcial
            text = getattr(chunk, "text", None)
            if text:
                yield text
