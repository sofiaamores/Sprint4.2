from typing import Optional

from pydantic import ValidationError

from core.api_client import APIClient
from models.linkedin_post import LinkedinPost


class Chatbot:
    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        self.api = APIClient(model=model)

    def ask(self, prompt: str, default: Optional[str] = None) -> str:
        value = input(prompt).strip()
        if not value and default is not None:
            return default
        return value

    def run_once(self) -> Optional[LinkedinPost]:
        print("\n=== Generador de posts para LinkedIn (Structured Output) ===")

        topic = self.ask("Tema del post: ")
        while not topic:
            print("El tema no puede estar vacío.")
            topic = self.ask("Tema del post: ")

        category = self.ask(
            "Categoría [programacion/marketing/legal] (default: programacion): ",
            default="programacion",
        ).lower()

        if category not in ("programacion", "marketing", "legal"):
            print("Categoría inválida. Usando 'programacion'.")
            category = "programacion"

        try:
            post = self.api.generate_linkedin_post(topic=topic, category=category)
            return post

        except ValidationError as e:
            # Error de Pydantic (estructura/tipos no válidos)
            print("\n❌ Error de validación (Pydantic). La salida no cumple el esquema.")
            print(str(e))
            return None

        except Exception as e:
            # Errores de API, refusal u otros problemas
            print("\n❌ Error al generar el post.")
            print(f"Detalle: {e}")
            return None

    @staticmethod
    def pretty_print(post: LinkedinPost) -> None:
        print("\n--- RESULTADO ---")
        print(f"\nTítulo:\n{post.title}")
        print(f"\nContenido:\n{post.content}")
        print("\nHashtags:")
        print(" ".join(post.hashtags))
        print(f"\nCategoría: {post.category}")
        print("-----------------\n")