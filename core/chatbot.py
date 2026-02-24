from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI

from core.rag_system import RAGSystem, RetrievedChunk


SYSTEM_PROMPT = """Eres un asistente interno de una empresa ficticia.
Reglas estrictas:
1) Responde ÚNICAMENTE usando la información disponible en el CONTEXTO proporcionado.
2) Si la respuesta no está en el contexto, di: "No lo sé con la información disponible en los documentos."
3) No inventes políticas, datos, nombres ni procedimientos.
4) Sé claro y directo.
"""


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


class Chatbot:
    """
    Flujo: pregunta -> retrieve -> construir contexto -> LLM -> respuesta
    Mantiene historial en memoria durante la sesión.
    """

    def __init__(
        self,
        rag: RAGSystem,
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_history_messages: int = 10,
    ) -> None:
        self.rag = rag
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.history: List[ChatMessage] = []
        self.max_history_messages = max_history_messages

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        parts = []
        for i, ch in enumerate(chunks, start=1):
            parts.append(
                f"[Fuente {i}: {ch.source}]\n{ch.content}"
            )
        return "\n\n---\n\n".join(parts)

    def _history_for_prompt(self) -> List[Dict[str, Any]]:
        """
        Limita el historial para no crecer infinito.
        """
        trimmed = self.history[-self.max_history_messages :]
        return [{"role": m.role, "content": m.content} for m in trimmed]

    def ask(self, user_question: str) -> str:
        chunks = self.rag.retrieve(user_question)
        context = self._build_context(chunks)

        # Si no hay contexto recuperado, forzamos "no lo sé" sin llamar al modelo (más estricto).
        if not context.strip():
            answer = "No lo sé con la información disponible en los documentos."
            self.history.append(ChatMessage(role="user", content=user_question))
            self.history.append(ChatMessage(role="assistant", content=answer))
            return answer

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history_for_prompt(),
            {
                "role": "user",
                "content": (
                    "CONTEXTO:\n"
                    f"{context}\n\n"
                    "PREGUNTA:\n"
                    f"{user_question}\n\n"
                    "INSTRUCCIÓN:\n"
                    "Responde solo con el contexto y, si no está, di la frase exacta indicada."
                ),
            },
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        # Guarda historial
        self.history.append(ChatMessage(role="user", content=user_question))
        self.history.append(ChatMessage(role="assistant", content=answer))

        return answer