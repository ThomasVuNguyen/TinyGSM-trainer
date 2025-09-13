import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("RIFT_API_KEY")
if not api_key:
    raise ValueError("RIFT_API_KEY not found in .env file. Please add RIFT_API_KEY=your_key_here to your .env file")

client = openai.OpenAI(
  api_key=api_key,
  base_url="https://inference.cloudrift.ai/v1"
)

completion = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-70B-Instruct-FP8",
  messages=[
    {"role": "user", "content": "What is the meaning of life?"}
  ],
  stream=True
)

for chunk in completion:
  print(chunk.choices[0].delta.content or "", end="")