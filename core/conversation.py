from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class ConversationState:
    """Mantiene el estado global de la conversación."""
    active_expert: str
    is_running: bool = True


@dataclass
class ConversationManager:
    """
    Gestiona el historial de conversación por experto.

    - Cada experto mantiene su propio contexto (lista de mensajes user/assistant).
    - El prompt 'system' se inyecta al construir el payload hacia el modelo.
    - Permite cambiar de experto manteniendo o reiniciando el historial.
    """
    system_prompts: Dict[str, str]
    default_expert: str
    max_messages_per_expert: int = 30

    _histories: Dict[str, List[Message]] = field(default_factory=dict)
    state: ConversationState = field(init=False)

    def __post_init__(self) -> None:
        if self.default_expert not in self.system_prompts:
            raise ValueError(
                f"default_expert '{self.default_expert}' no existe en system_prompts."
            )
        self.state = ConversationState(active_expert=self.default_expert)
        # Inicializa el historial vacío del experto por defecto
        self._histories.setdefault(self.default_expert, [])

    # -----------------------------
    # Estado / experto activo
    # -----------------------------
    def get_active_expert(self) -> str:
        return self.state.active_expert

    def set_active_expert(self, expert_key: str, keep_history: bool = True) -> None:
        """
        Cambia el experto activo.
        - keep_history=True: conserva historial existente de ese experto
        - keep_history=False: reinicia historial de ese experto
        """
        if expert_key not in self.system_prompts:
            raise ValueError(f"Experto desconocido: '{expert_key}'")

        self.state.active_expert = expert_key

        if not keep_history:
            self._histories[expert_key] = []
        else:
            self._histories.setdefault(expert_key, [])

    # -----------------------------
    # Historial
    # -----------------------------
    def add_user_message(self, content: str, expert_key: Optional[str] = None) -> None:
        expert = expert_key or self.get_active_expert()
        self._ensure_expert(expert)
        self._histories[expert].append({"role": "user", "content": content})
        self._trim(expert)

    def add_assistant_message(self, content: str, expert_key: Optional[str] = None) -> None:
        expert = expert_key or self.get_active_expert()
        self._ensure_expert(expert)
        self._histories[expert].append({"role": "assistant", "content": content})
        self._trim(expert)

    def reset_history(self, expert_key: Optional[str] = None) -> None:
        """Reinicia el historial del experto indicado (o del activo si no se indica)."""
        expert = expert_key or self.get_active_expert()
        self._ensure_expert(expert)
        self._histories[expert] = []

    def reset_all(self) -> None:
        """Reinicia todos los historiales y vuelve al experto por defecto."""
        self._histories = {k: [] for k in self.system_prompts.keys()}
        self.state.active_expert = self.default_expert

    def get_history(self, expert_key: Optional[str] = None) -> List[Message]:
        """Devuelve copia del historial (sin el system prompt)."""
        expert = expert_key or self.get_active_expert()
        self._ensure_expert(expert)
        return list(self._histories[expert])

    # -----------------------------
    # Payload para el modelo
    # -----------------------------
    def build_messages_for_model(self, expert_key: Optional[str] = None) -> List[Message]:
        """
        Devuelve la lista de mensajes para enviar a Ollama:
        - 1º: system prompt del experto
        - Luego: historial user/assistant del experto
        """
        expert = expert_key or self.get_active_expert()
        self._ensure_expert(expert)

        system_prompt = self.system_prompts[expert]
        return [{"role": "system", "content": system_prompt}, *self._histories[expert]]

    # -----------------------------
    # Control de ejecución
    # -----------------------------
    def stop(self) -> None:
        self.state.is_running = False

    # -----------------------------
    # Helpers internos
    # -----------------------------
    def _ensure_expert(self, expert_key: str) -> None:
        if expert_key not in self.system_prompts:
            raise ValueError(f"Experto desconocido: '{expert_key}'")
        self._histories.setdefault(expert_key, [])

    def _trim(self, expert_key: str) -> None:
        """
        Recorta el historial del experto para no crecer indefinidamente.
        Mantiene los últimos max_messages_per_expert mensajes (user/assistant).
        """
        hist = self._histories.get(expert_key, [])
        if len(hist) > self.max_messages_per_expert:
            self._histories[expert_key] = hist[-self.max_messages_per_expert:]
