from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from experts.expert_prompts import EXPERTS, DEFAULT_EXPERT_KEY, get_expert_label, list_experts
from core.conversation import ConversationManager
from core.chatbot import ExpertChatbot, ModelNotAvailableError, OllamaConnectionError

console = Console()


def show_header(active_expert: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]Chatbot de expertos (offline)[/bold]\n"
            f"Experto activo: [cyan]{get_expert_label(active_expert)}[/cyan]\n\n"
            f"[dim]Comandos: /menu  /expert  /reset  /exit[/dim]",
            title="Ollama + gemma3:1b",
        )
    )


def choose_expert_menu(current: str) -> str:
    console.print("\n[bold]Selecciona un experto temático:[/bold]")
    keys = list_experts()
    for i, k in enumerate(keys, start=1):
        marker = " (activo)" if k == current else ""
        console.print(f"  [bold]{i}[/bold]. {get_expert_label(k)} [dim]/{k}[/dim]{marker}")

    while True:
        choice = Prompt.ask("\nIntroduce número (1-3) o clave (programacion/marketing/juridico)", default=current)
        choice = choice.strip().lower()

        # Permite número
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
            console.print("[red]Opción numérica inválida.[/red]")
            continue

        # Permite clave directa
        if choice in EXPERTS:
            return choice

        console.print("[red]Opción inválida. Prueba otra vez.[/red]")


def main() -> None:
    # 1) Construir ConversationManager con prompts de expertos
    conversation = ConversationManager(
        system_prompts={k: v["system"] for k, v in EXPERTS.items()},
        default_expert=DEFAULT_EXPERT_KEY,
        max_messages_per_expert=30,
    )

    # 2) Crear chatbot
    bot = ExpertChatbot(conversation)

    # 3) Health check + mensajes informativos
    console.print("[dim]Comprobando servicio local de Ollama y modelo...[/dim]")
    try:
        bot.health_check()
    except ModelNotAvailableError as e:
        console.print(f"[red]{e}[/red]")
        return
    except OllamaConnectionError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print("[green]OK:[/green] Ollama accesible y modelo disponible.\n")

    # 4) Menú inicial
    active = conversation.get_active_expert()
    show_header(active)
    active = choose_expert_menu(active)
    conversation.set_active_expert(active, keep_history=True)

    console.print(
        Panel(
            f"Has seleccionado: [cyan]{get_expert_label(active)}[/cyan]\n"
            f"[dim]Escribe tu mensaje y pulsa Enter. Usa /menu para cambiar de experto.[/dim]",
            title="Listo",
        )
    )

    # 5) Bucle conversación
    while conversation.state.is_running:
        active = conversation.get_active_expert()
        user_text = Prompt.ask(f"\n[bold][{active}][/bold] Tú").strip()

        if not user_text:
            console.print("[dim]Mensaje vacío. Escribe algo o usa /exit para salir.[/dim]")
            continue

        # ---- comandos ----
        cmd = user_text.lower()

        if cmd in ("/exit", "/salir", "salir"):
            if Confirm.ask("¿Quieres salir del chatbot?", default=True):
                conversation.stop()
            continue

        if cmd in ("/reset", "/reiniciar"):
            if Confirm.ask("¿Reiniciar el historial del experto activo?", default=False):
                conversation.reset_history()
                console.print("[yellow]Historial reiniciado para el experto activo.[/yellow]")
            continue

        if cmd in ("/menu", "/expert"):
            new_expert = choose_expert_menu(active)
            keep = Confirm.ask("¿Mantener el historial de ese experto?", default=True)
            conversation.set_active_expert(new_expert, keep_history=keep)
            console.print(f"[cyan]Experto activo ahora:[/cyan] {get_expert_label(new_expert)}")
            continue

        # ---- conversación normal ----
        try:
            answer = bot.ask(user_text)
        except OllamaConnectionError as e:
            console.print(f"[red]{e}[/red]")
            console.print("[dim]Tip: verifica que Ollama está abierto y en ejecución.[/dim]")
            continue
        except Exception as e:
            console.print(f"[red]Error inesperado:[/red] {e}")
            continue

        console.print(Panel(answer, title=f"Respuesta ({get_expert_label(active)})", border_style="green"))


if __name__ == "__main__":
    main()
