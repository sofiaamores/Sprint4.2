import os
from dotenv import load_dotenv
from openai import OpenAI

from models.linkedin_post import LinkedinPost


SYSTEM_INSTRUCTIONS = """Eres un asistente experto en escribir posts de LinkedIn en español.
Genera un post profesional, claro y útil. Evita exageraciones y emojis en exceso.
Devuelve contenido listo para publicar.
La salida debe ajustarse EXACTAMENTE al esquema solicitado (title, content, hashtags, category).
"""


class APIClient:
    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en el archivo .env")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_linkedin_post(self, topic: str, category: str) -> LinkedinPost:
        prompt = (
            f"Tema del post: {topic}\n"
            f"Categoría (obligatoria): {category}\n\n"
            "Requisitos:\n"
            "- 1 gancho inicial + 2-4 párrafos cortos\n"
            "- 3 a 10 hashtags relevantes\n"
        )

        response = self.client.responses.parse(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=prompt,
            text_format=LinkedinPost,
        )

        # 1) Si el modelo rechaza (refusal), no habrá objeto parseado.
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            # Intentamos sacar un texto útil para mostrar al usuario
            text = getattr(response, "output_text", "") or "El modelo rechazó la solicitud."
            raise RuntimeError(f"Refusal o salida no parseable: {text}")

        return parsed