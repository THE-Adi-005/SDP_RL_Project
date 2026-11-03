import os
import time
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency: float
    error: str = ""

class LLMClient:
    """
    [SDP: FACADE / FACTORY ABSTRACTION]
    This class abstracts the complex initialization of the OpenAI client.
    By reading configurations at runtime, it acts as a simple Factory for 
    different underlying LLM models (e.g., Qwen 7b vs 3b, or OpenAI API) 
    without requiring the core reasoning system to understand connection mechanics.
    """
    def __init__(self, base_url=None, api_key=None, model=None,
                 temperature=0.2, max_tokens=512, timeout=60):
        self.client = OpenAI(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        self.model = model or os.getenv("MODEL_NAME", "qwen2.5:7b-instruct-q4_K_M")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str) -> LLMResponse:
        start = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            latency = time.time() - start
            text = resp.choices[0].message.content
            usage = resp.usage
            return LLMResponse(
                text=text,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                latency=latency,
            )
        except Exception as e:
            return LLMResponse(text="", input_tokens=0, output_tokens=0,
                                latency=time.time() - start, error=str(e))
