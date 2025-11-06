"""OpenAI LLM client for structured extraction and generation"""

import json
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

            if response_format == "json":
                return json.loads(content)
            else:
                return {"text": content}

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
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

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
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
            results.append(result)
        return results
