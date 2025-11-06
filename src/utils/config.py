"""Configuration management"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration settings for the News Fact Validation system"""

    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

    # Temperature settings for different tasks
    EXTRACTION_TEMPERATURE = 0.1  # Low temperature for structured extraction
    GENERATION_TEMPERATURE = 0.7  # Higher for creative narrative generation

    # Graph settings
    GRAPH_CONFIDENCE_THRESHOLD = 0.5

    # GNN settings
    GNN_HIDDEN_DIM = 64
    GNN_NUM_LAYERS = 2
    GNN_DROPOUT = 0.1

    # Output directories
    OUTPUT_DIR = "outputs"
    GRAPH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "graphs")

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found. Please set it in .env file"
            )

    @classmethod
    def create_output_dirs(cls):
        """Create output directories if they don't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.GRAPH_OUTPUT_DIR, exist_ok=True)
