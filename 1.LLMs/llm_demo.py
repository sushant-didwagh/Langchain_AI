import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from groq_llm_helper import get_groq_llm

llm = get_groq_llm()
result = llm.invoke("What is the capital of India")
print(result)