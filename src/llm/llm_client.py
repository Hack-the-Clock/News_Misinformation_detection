"""OpenAI LLM client for structured extraction and generation"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from openai import OpenAI

from ..utils.config import Config


class LLMClient:
    """Client for interacting with OpenAI's API"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM client

        Args:
            api_key: OpenAI API key (defaults to Config.OPENAI_API_KEY)
            model: Model name (defaults to Config.OPENAI_MODEL)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        self.client = OpenAI(api_key=self.api_key)

    def _log_llm_call(self, method: str, payload: dict, response: dict):
        """
        Save LLM call details to a log file for tracing and future use.
        """
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "llm_calls.log")
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "payload": payload,
            "response": response
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def extract_structured_data(
        self,
        text: str,
        system_prompt: str,
        response_format: str = "json",
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text using OpenAI

        Args:
            text: Input text to analyze
            system_prompt: System prompt defining the extraction task
            response_format: Expected format ("json" or "text")
            temperature: Sampling temperature (defaults to Config.EXTRACTION_TEMPERATURE)

        Returns:
            Parsed JSON response or text response
        """
        if temperature is None:
            temperature = Config.EXTRACTION_TEMPERATURE

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"} if response_format == "json" else None
            )

            content = response.choices[0].message.content
            result = json.loads(content) if response_format == "json" else {"text": content}
            self._log_llm_call(
                method="extract_structured_data",
                payload={"text": text, "system_prompt": system_prompt, "response_format": response_format, "temperature": temperature},
                response=result
            )
            return result

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            self._log_llm_call(
                method="extract_structured_data",
                payload={"text": text, "system_prompt": system_prompt, "response_format": response_format, "temperature": temperature},
                response={"error": str(e)}
            )
            return {}

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = None
    ) -> str:
        """
        Generate text using OpenAI

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (defaults to Config.GENERATION_TEMPERATURE)

        Returns:
            Generated text
        """
        if temperature is None:
            temperature = Config.GENERATION_TEMPERATURE

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )

            result = response.choices[0].message.content
            self._log_llm_call(
                method="generate_text",
                payload={"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature},
                response={"text": result}
            )
            return result

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            self._log_llm_call(
                method="generate_text",
                payload={"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature},
                response={"error": str(e)}
            )
            return ""

    def batch_extract(
        self,
        texts: List[str],
        system_prompt: str,
        response_format: str = "json"
    ) -> List[Dict[str, Any]]:
        """
        Extract structured data from multiple texts

        Args:
            texts: List of input texts
            system_prompt: System prompt for extraction
            response_format: Expected format

        Returns:
            List of extracted data
        """
        results = []
        for text in texts:
            result = self.extract_structured_data(text, system_prompt, response_format)
            # Each extract_structured_data call is already logged
            results.append(result)
        return results
