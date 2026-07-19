import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from groq_llm_helper import get_chat_groq_model
from dotenv import load_dotenv

load_dotenv()

model = get_chat_groq_model(model='llama-3.3-70b-versatile', temperature=1.5, max_completion_tokens=10)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
