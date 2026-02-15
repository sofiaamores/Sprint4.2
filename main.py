# main.py
import time
from dotenv import load_dotenv

from core.conversation import Conversation
from core.chatbot import Chatbot

from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider


load_dotenv()


def stream_print(text_iter):
    """Imprime chunks en streaming y devuelve el texto completo."""
    full = []
    for chunk in text_iter:
        if not chunk:
            continue
        full.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(full)


def main():
    # Prompt del sistema (opcional pero recomendable)
    system_prompt = (
        "Eres un asistente útil y conciso. "
        "Si no sabes algo, dilo y pide contexto."
    )

    conversation = Conversation(system_prompt=system_prompt)

    # Orden de fallback: OpenAI → Anthropic → Gemini
    providers = [OpenAIProvider(), AnthropicProvider(), GeminiProvider()]
    bot = Chatbot(providers)

    print("Chatbot con fallback (OpenAI → Anthropic → Gemini)")
    print("Comandos: /salir, /reset\n")

    last_provider = None

    while True:
        user_input = input("Tú: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "/salir":
            print("Saliendo…")
            break

        if user_input.lower() == "/reset":
            conversation.clear()
            conversation.add_system(system_prompt)
            last_provider = None
            print("Historial reiniciado.\n")
            continue

        # 1) guardamos turno usuario
        conversation.add_user(user_input)

        try:
            # 2) pedimos respuesta (stream) con fallback
            stream, provider_name = bot.ask(conversation.messages())

            if last_provider and provider_name != last_provider:
                print(f"\n[INFO] Cambiando proveedor: {last_provider} → {provider_name}\n")

            print(f"{provider_name}: ", end="", flush=True)
            assistant_text = stream_print(stream)

            # 3) guardamos turno asistente
            conversation.add_assistant(assistant_text)
            last_provider = provider_name

        except Exception as e:
            print("\n[ERROR] No se pudo generar respuesta con ningún proveedor.")
            print(f"[ERROR] Detalle: {e}\n")

            # Si falla, no añadimos un assistant vacío: mantenemos el historial solo con el user
            # (si prefieres revertir el último user, dime y lo ajusto)

            # Pausa corta por si hay rate-limits encadenados
            time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())