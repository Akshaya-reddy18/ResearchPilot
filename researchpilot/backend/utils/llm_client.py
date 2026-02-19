import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)


class GroqClient:
    # Default model - can be overridden via environment variable
    # Updated to llama-3.3-70b-versatile (llama-3.1-70b-versatile was deprecated Jan 24, 2025)
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment. Please add it to your .env file.")
        
        # Check if key looks valid (Groq keys typically start with 'gsk_')
        if not api_key.startswith('gsk_'):
            logger.warning(f"API key format may be incorrect. Groq keys typically start with 'gsk_'")
        
        # Get model from environment or use default
        self.model = os.getenv("GROQ_MODEL", self.DEFAULT_MODEL)
        logger.info(f"Using Groq model: {self.model}")
        
        try:
            # Initialize Groq client - only pass api_key to avoid proxy issues
            # Some environments may have HTTP_PROXY/HTTPS_PROXY set which can cause issues
            self.client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully")
        except TypeError as e:
            # Handle version compatibility issues
            if "proxies" in str(e) or "unexpected keyword argument" in str(e):
                logger.warning("Groq SDK version may have compatibility issues. Trying alternative initialization...")
                # Try without any extra arguments
                try:
                    self.client = Groq(api_key=api_key)
                    logger.info("Groq client initialized successfully (alternative method)")
                except Exception as e2:
                    raise RuntimeError(f"Failed to initialize Groq client: {str(e2)}. Please check your API key and Groq SDK version.")
            else:
                raise RuntimeError(f"Failed to initialize Groq client: {str(e)}. Please check your API key.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {str(e)}. Please check your API key.")

    def generate_response(self, prompt: str) -> str:
        # Use the model specified in initialization (default: llama-3.3-70b-versatile)
        # Alternative models: llama-3.1-8b-instant, llama-3.3-70b-specdec, qwen/qwen3-32b
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an advanced research analysis AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        # Response structure returned by Groq SDK
        try:
            analysis = response.choices[0].message.content.strip()
            return analysis
        except Exception:
            # Fallback to string representation
            return str(response)
