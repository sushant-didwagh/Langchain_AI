import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from groq_llm_helper import get_chat_groq_model

load_dotenv()

model = get_chat_groq_model(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_completion_tokens=80,
)

result = model.invoke("capital of india")
print(result.content)