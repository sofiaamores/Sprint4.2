from core.chatbot import Chatbot


def main() -> None:
    bot = Chatbot(model="gpt-4o-2024-08-06")

    print("LinkedIn Post Generator (escribe 'q' para salir)\n")

    while True:
        cmd = input("Pulsa ENTER para generar un post o escribe 'q' para salir: ").strip().lower()
        if cmd == "q":
            print("Saliendo. 👋")
            break

        post = bot.run_once()
        if post is not None:
            bot.pretty_print(post)


if __name__ == "__main__":
    main()
