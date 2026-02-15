class Conversation:
    def __init__(self, system_prompt: str | None = None):
        self._messages = []

        if system_prompt:
            self.add_system(system_prompt)

    def add_system(self, content: str):
        self._messages.append({
            "role": "system",
            "content": content
        })

    def add_user(self, content: str):
        self._messages.append({
            "role": "user",
            "content": content
        })

    def add_assistant(self, content: str):
        self._messages.append({
            "role": "assistant",
            "content": content
        })

    def messages(self):
        return self._messages

    def clear(self):
        self._messages = []
