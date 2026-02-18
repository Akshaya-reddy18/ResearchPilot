import os
from groq import Groq


class GroqClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an advanced research analysis AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Response structure returned by Groq SDK
        try:
            return response.choices[0].message.content
        except Exception:
            # Fallback to string representation
            return str(response)
