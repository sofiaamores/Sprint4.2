from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ollama

from core.conversation import ConversationManager


class OllamaConnectionError(RuntimeError):
    """Error al conectar con el servicio local de Ollama."""


class ModelNotAvailableError(RuntimeError):
    """El modelo requerido no está disponible en Ollama."""


@dataclass
class ChatbotConfig:
    model: str = "gemma3:1b"
    temperature: float = 0.3
    num_predict: int = 400  # límite aproximado de tokens de salida


class ExpertChatbot:
    """
    Lógica principal del chatbot:
    - Envía mensajes a Ollama (offline, servicio local).
    - Usa el ConversationManager para mantener historial por experto.
    - Maneja errores de conexión y de modelo no disponible.
    """

    def __init__(self, conversation: ConversationManager, config: Optional[ChatbotConfig] = None) -> None:
        self.conversation = conversation
        self.config = config or ChatbotConfig()

    # -----------------------------
    # Checks (conexión y modelo)
    # -----------------------------
    def health_check(self) -> None:
        """
        Verifica que Ollama está accesible y que el modelo existe.
        Lanza excepciones claras si algo falla.
        """
        try:
            models = ollama.list()
        except Exception as e:
            raise OllamaConnectionError(
                "No se pudo conectar con Ollama. "
                "Asegúrate de que el servicio esté instalado y ejecutándose."
            ) from e

        available = {m.get("model") for m in models.get("models", [])}
        if self.config.model not in available:
            raise ModelNotAvailableError(
                f"El modelo '{self.config.model}' no está disponible en Ollama.\n"
                f"Solución: ejecuta `ollama pull {self.config.model}` y vuelve a intentar."
            )

    # -----------------------------
    # Chat
    # -----------------------------
    def ask(self, user_text: str) -> str:
        """
        Envía el mensaje del usuario al modelo con el contexto del experto activo.
        Devuelve la respuesta del asistente (string).
        """
        # Guardar input del usuario en el historial del experto activo
        self.conversation.add_user_message(user_text)

        # Construir mensajes (system + historial) para el modelo
        messages = self.conversation.build_messages_for_model()

        try:
            resp = ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.num_predict,
                },
            )
        except Exception as e:
            # Si falla aquí, normalmente es servicio caído o modelo no accesible
            raise OllamaConnectionError(
                "Error llamando a Ollama. Revisa que el servicio esté activo y el modelo descargado."
            ) from e

        # Extraer texto
        assistant_text = (resp.get("message") or {}).get("content", "").strip()
        if not assistant_text:
            assistant_text = "No he podido generar una respuesta. Intenta reformular la pregunta."

        # Guardar respuesta en historial
        self.conversation.add_assistant_message(assistant_text)

        return assistant_text
