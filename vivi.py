##### LIBRARIES #####
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_ollama import OllamaLLM
from vivi_character import VIVICharacter
from datetime import datetime
import json
import os



##### TEST-LIBRARIES #####


##### INPUT VARIABLES SETTINGS #####
# reasoning_model = 'gemma3:27b'
reasoning_model = 'llama3.1'
# reasoning_model = 'benevolentjoker/nsfwmonika'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# memory_file = "memory/VIVI-memory.json"
# character_file = "characters/VIVI-character.json"

##### LLM PROMPTS #####
def create_prompt(format_instructions):
    QA_TEMPLATE = """
        {format_instructions}
     
    """
    return PromptTemplate(
        input_variables=[""], 
        partial_variables={"format_instructions": format_instructions},
        template=QA_TEMPLATE)


##### LLM VARIABLES SETTINGS #####
# output_parser = JsonOutputParser()
# format_instructions = output_parser.get_format_instructions()
# reasoning_model_list =["nemotron-mini"] 


##### FUNCTIONS #####
def load_memory(filepath="memory/VIVI-memory.json"):
    try:
        with open(filepath, "rb+") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting with empty memory.")
        return []

def load_character(filepath="characters/VIVI-character.json"):
    try:
        with open(filepath, "rb+") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting with default character.")
        return {}

def run_live_character(memory, character_data):
    character = VIVICharacter(character_data)
    print(character.intro())

    # Load model
    llm = OllamaLLM(model=reasoning_model, temperature=0.1)

    # Prompt Template
    prompt_template = PromptTemplate.from_template("""
      You are {name}, a friendly AI assistant with the following personality traits:
      {personality}

      Your recent memories include:
      {memory}

      User: {user_input}
      AI:
    """)

    # Build prompt chain
    chain = (
        RunnableMap({
            "name": lambda x: x["character"].name,
            "personality": lambda x: x["character"].personality_summary(),
            "memory": lambda x: "\n".join(
                f"{m['timestamp']}: {m['content']}" for m in x["memory"][-5:]
            ),
            "user_input": lambda x: x["user_input"]
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Live loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "!stop"):
            print(character.outro())
            break

        try:
            response = chain.invoke({
                "character": character,
                "memory": memory,
                "user_input": user_input
            })
        except Exception as e:
            response = f"(Oops, something went wrong: {str(e)})"

        if not response.strip():
            response = character.default_response

        print(f"{character.name}: {response}")

        memory.append({
            "timestamp": datetime.now().isoformat(),
            "content": f"User: {user_input}\n{character.name}: {response}"
        })

        with open(os.path.join(BASE_DIR, "memory/VIVI-memory.json"), "w") as f:
          json.dump(memory, f, indent=4)


###---------------------------------------------------------------###
if __name__ == "__main__": 
    start_time = datetime.now()
    print("Importing vivi...", flush = True)
    memory = load_memory()
    character = load_character()
    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    # print(seconds, flush = True)

    run_live_character(memory, character)
