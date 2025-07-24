class VIVICharacter:
    def __init__(self, data: dict):
        self.name = data.get("name", "VIVI")
        self.description = data.get("description", "")
        self.personality_traits = data.get("personality_traits", [])
        self.greeting = data.get("greeting", "Hello! I'm VIVI.")
        self.farewell = data.get("farewell", "Bye for now!")
        self.default_response = data.get("default_response", "Hmm, interesting question...")
        self.knowledge_domain = data.get("knowledge_domain", [])
        self.agile_reminders = data.get("agile_reminders", [])

    def personality_summary(self) -> str:
        return "; ".join(
            f"{t['trait']}: {t['description']}" for t in self.personality_traits
        )

    def intro(self) -> str:
        return self.greeting

    def outro(self) -> str:
        return self.farewell

    def random_reminder(self) -> str:
        import random
        return random.choice(self.agile_reminders) if self.agile_reminders else ""
