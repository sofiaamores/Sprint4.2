# main.py
from __future__ import annotations

import sys
from dotenv import load_dotenv

from core.rag_system import RAGSystem
from core.chatbot import Chatbot


def print_banner() -> None:
    print("=" * 60)
    print(" Chatbot RAG (CLI) — Empresa ficticia (Markdown)")
    print("=" * 60)
    print("Comandos:")
    print("  /salir, quit, exit  -> cerrar")
    print("  /help              -> ayuda")
    print("=" * 60)
    print()


def main() -> None:
    # 1) Cargar variables de entorno (.env)
    load_dotenv()

    # 2) Inicializar e ingestar documentos
    try:
        rag = RAGSystem(documents_path="documents", k=4)
        rag.ingest()
    except Exception as e:
        print("\n[ERROR] No se pudo inicializar el sistema RAG.")
        print(f"Motivo: {e}\n")
        print("Revisa que:")
        print("- Existe la carpeta /documents")
        print("- Hay al menos 2 archivos .md dentro")
        print("- Tienes OPENAI_API_KEY en el archivo .env")
        sys.exit(1)

    # 3) Crear chatbot
    try:
        bot = Chatbot(
            rag=rag,
            model_name="gpt-4o-2024-08-06",  # o simplemente "gpt-4o"
            temperature=0.2
        )
    except Exception as e:
        print("\n[ERROR] No se pudo inicializar el modelo de chat.")
        print(f"Motivo: {e}\n")
        sys.exit(1)

    # 4) Mensajes informativos
    print_banner()
    print("Sistema listo ✅")
    print("- Embeddings: text-embedding-3-small")
    print("- VectorStore: InMemoryVectorStore (en memoria)")
    print("- Modelo chat: gpt-4o")
    print("- Respuestas SOLO basadas en documentos.\n")

    # 5) Bucle principal
    while True:
        try:
            user_text = input("Tú: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSaliendo...")
            break

        if not user_text:
            continue

        cmd = user_text.lower().strip()
        if cmd in {"/salir", "salir", "quit", "exit"}:
            print("Saliendo...")
            break

        if cmd == "/help":
            print("\nAyuda:")
            print("- Escribe una pregunta sobre la empresa ficticia.")
            print("- Usa /salir o quit para cerrar.")
            print("- Si no hay evidencia en los documentos, responderá 'No lo sé...'\n")
            continue

        # 6) Manejo de errores al consultar al modelo
        try:
            answer = bot.ask(user_text)
            print(f"\nBot: {answer}\n")
        except Exception as e:
            print("\n[ERROR] Falló la consulta al modelo o la conexión.")
            print(f"Motivo: {e}\n")
            print("Sugerencias:")
            print("- Verifica tu OPENAI_API_KEY en .env")
            print("- Revisa conexión a Internet")
            print("- Prueba de nuevo en unos segundos\n")


if __name__ == "__main__":
    main()