class Chatbot:
    def __init__(self, providers):
        self.providers = providers
        self.active_provider = None

    def _should_fallback(self, exception: Exception):
        msg = str(exception).lower()

        if "429" in msg or "rate limit" in msg:
            return True

        if "timeout" in msg or "connection" in msg:
            return True

        if "5" in msg:  # errores 5xx
            return True

        if "401" in msg or "unauthorized" in msg:
            return True

        return True  # robustez en ejercicio

    def ask(self, messages):
        last_error = None

        for provider in self.providers:
            try:
                return provider.stream(messages), provider.name
            except Exception as e:
                last_error = e
                if not self._should_fallback(e):
                    break
                continue

        raise RuntimeError(f"Todos los proveedores fallaron. Último error: {last_error}")