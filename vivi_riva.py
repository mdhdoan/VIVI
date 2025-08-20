"""Live animated chat interface using NVIDIA Riva for speech services."""

import json
import os
import threading
from datetime import datetime

import numpy as np
import pygame
import sounddevice as sd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_ollama import OllamaLLM
from riva.client import (
    ASRService,
    AudioEncoding,
    Auth,
    RecognitionConfig,
    SpeechSynthesisService,
)

from vivi_character import VIVICharacter

# ---------- SETTINGS ----------
RIVA_URL = os.getenv("RIVA_SPEECH_API_URL", "localhost:50051")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REASONING_MODEL = "llama3.1"

# ---------- DATA LOADING ----------

def load_memory(filepath: str = os.path.join(BASE_DIR, "memory/VIVI-memory.json")):
    try:
        with open(filepath, "rb+") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting with empty memory.")
        return []

def load_character(filepath: str = os.path.join(BASE_DIR, "characters/VIVI-character.json")):
    try:
        with open(filepath, "rb+") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting with default character.")
        return {}

# ---------- AUDIO HELPERS ----------

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Record audio from the default microphone and return raw bytes."""
    print("Listening...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    return audio.tobytes()

def transcribe_audio(asr: ASRService, audio_bytes: bytes, sample_rate: int = 16000) -> str:
    """Send audio bytes to Riva ASR and return the transcript."""
    config = RecognitionConfig(
        encoding=AudioEncoding.LINEAR_PCM,
        sample_rate_hertz=sample_rate,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    response = asr.offline_recognize(audio_bytes, config)
    if response.results:
        return response.results[0].alternatives[0].transcript.strip()
    return ""

def play_audio_with_animation(audio: np.ndarray, sample_rate: int, screen: pygame.Surface, clock: pygame.time.Clock):
    """Play audio samples while toggling the avatar mouth for a simple animation."""
    def _play():
        sd.play(audio, sample_rate)
        sd.wait()

    thread = threading.Thread(target=_play)
    thread.start()
    mouth_open = False
    while thread.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sd.stop()
                pygame.quit()
                return
        mouth_open = not mouth_open
        draw_avatar(screen, mouth_open)
        clock.tick(6)
    draw_avatar(screen, False)

# ---------- VISUALS ----------

def draw_avatar(screen: pygame.Surface, mouth_open: bool):
    screen.fill((30, 30, 30))
    pygame.draw.circle(screen, (255, 224, 189), (200, 200), 100)
    mouth_rect = pygame.Rect(160, 240 if mouth_open else 250, 80, 30 if mouth_open else 10)
    pygame.draw.rect(screen, (150, 0, 0), mouth_rect)
    pygame.display.flip()

# ---------- MAIN CHAT LOOP ----------

def run_chat():
    memory = load_memory()
    character = VIVICharacter(load_character())
    print(character.intro())

    auth = Auth(uri=RIVA_URL)
    asr = ASRService(auth)
    tts = SpeechSynthesisService(auth)

    llm = OllamaLLM(model=REASONING_MODEL, temperature=0.1)
    prompt_template = PromptTemplate.from_template(
        """
      You are {name}, a friendly AI assistant with the following personality traits:
      {personality}

      Your recent memories include:
      {memory}

      User: {user_input}
      AI:
    """
    )
    chain = (
        RunnableMap(
            {
                "name": lambda x: x["character"].name,
                "personality": lambda x: x["character"].personality_summary(),
                "memory": lambda x: "\n".join(
                    f"{m['timestamp']}: {m['content']}" for m in x["memory"][-5:]
                ),
                "user_input": lambda x: x["user_input"],
            }
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()
    draw_avatar(screen, False)

    while True:
        audio_bytes = record_audio()
        user_input = transcribe_audio(asr, audio_bytes)
        if not user_input:
            continue
        print(f"You: {user_input}")
        if user_input.lower() in ("exit", "!stop"):
            print(character.outro())
            break

        try:
            response = chain.invoke(
                {
                    "character": character,
                    "memory": memory,
                    "user_input": user_input,
                }
            )
        except Exception as e:
            response = f"(Oops, something went wrong: {e})"

        print(f"{character.name}: {response}")

        tts_response = tts.synthesize(
            response,
            voice_name="English-US-Female-1",
            sample_rate_hz=22050,
            language_code="en-US",
        )
        samples = np.frombuffer(tts_response.audio, dtype=np.int16)
        play_audio_with_animation(samples, tts_response.sample_rate_hz, screen, clock)

        memory.append(
            {
                "timestamp": datetime.now().isoformat(),
                "content": f"User: {user_input}\n{character.name}: {response}",
            }
        )
        with open(os.path.join(BASE_DIR, "memory/VIVI-memory.json"), "w") as f:
            json.dump(memory, f, indent=4)

    pygame.quit()

if __name__ == "__main__":
    run_chat()
