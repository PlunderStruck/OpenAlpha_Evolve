# config/settings.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Gemini API Configuration (Optional - Comment out if not using) ---
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     print("Warning: GEMINI_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder.")
#     GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Placeholder

# --- LM Studio Configuration ---
# Fetches the base URL for the LM Studio API from environment variables.
# Defaults to "http://localhost:1234/v1" if not set.
LMSTUDIO_API_BASE_URL = os.getenv("LMSTUDIO_API_BASE_URL", "http://localhost:1234/v1")

# Fetches the model name for LM Studio. This might be a placeholder,
# as the actual model used is often selected within the LM Studio application itself.
# Defaults to "local-model" if not set.
LMSTUDIO_MODEL_NAME = os.getenv("LMSTUDIO_MODEL_NAME", "local-model")

# API key for LM Studio (if your server is configured to require one).
# Typically, the local LM Studio server doesn't require an API key by default.
# LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY")


# --- Original Gemini Model Names (Comment out or remove if only using LM Studio) ---
# GEMINI_PRO_MODEL_NAME = "gemini-1.5-flash-latest" # Example model
# GEMINI_FLASH_MODEL_NAME = "gemini-1.5-flash-latest" # Example model
# GEMINI_EVALUATION_MODEL = "gemini-1.5-flash-latest" # Example model for evaluation tasks


# --- Evolutionary Algorithm Parameters ---
POPULATION_SIZE = int(os.getenv("POPULATION_SIZE", 5))  # Number of individuals in each generation
GENERATIONS = int(os.getenv("GENERATIONS", 2))          # Total number of generations to run
ELITISM_COUNT = int(os.getenv("ELITISM_COUNT", 1))      # Number of best individuals to carry over directly to the next generation
MUTATION_RATE = float(os.getenv("MUTATION_RATE", 0.7))  # Probability of an individual undergoing mutation
CROSSOVER_RATE = float(os.getenv("CROSSOVER_RATE", 0.2)) # Probability of performing crossover (currently less emphasized with LLM diffs)

# --- Evaluation Parameters ---
# Timeout in seconds for executing a program during evaluation.
EVALUATION_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_TIMEOUT_SECONDS", 800))

# --- Database Configuration ---
# Type of database to use ("in_memory" or potentially "sqlite" if implemented)
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "in_memory")
# Path for the database file (relevant if not using in-memory)
DATABASE_PATH = os.getenv("DATABASE_PATH", "program_database.json") # Or .sqlite for SQLite

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # Logging level (e.g., DEBUG, INFO, WARNING)
LOG_FILE = os.getenv("LOG_FILE", "alpha_evolve.log")   # Name of the log file

# --- API Interaction Parameters ---
# Maximum number of retries for API calls (e.g., to LLM)
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", 5))
# Initial delay in seconds between API call retries (will likely use exponential backoff)
API_RETRY_DELAY_SECONDS = int(os.getenv("API_RETRY_DELAY_SECONDS", 10))

# --- Reinforcement Learning Fine-Tuner Parameters (Placeholders) ---
# Interval (in generations) at which RL training might occur
RL_TRAINING_INTERVAL_GENERATIONS = int(os.getenv("RL_TRAINING_INTERVAL_GENERATIONS", 50))
# Path to save/load the RL fine-tuner model
RL_MODEL_PATH = os.getenv("RL_MODEL_PATH", "rl_finetuner_model.pth")

# --- Monitoring Parameters (Placeholders) ---
# URL for a potential monitoring dashboard
MONITORING_DASHBOARD_URL = os.getenv("MONITORING_DASHBOARD_URL", "http://localhost:8080")


# --- Utility Functions for Settings ---
def get_setting(key, default=None):
    """
    Retrieves a setting value from the current module's global scope.

    Args:
        key (str): The name of the setting (variable name).
        default (Any, optional): The default value to return if the key is not found.

    Returns:
        Any: The value of the setting or the default.
    """
    return globals().get(key, default)

def get_llm_model_endpoint(provider: str = "lmstudio"):
    """
    Returns the appropriate model endpoint or identifier based on the provider.

    Args:
        provider (str): The LLM provider, e.g., "lmstudio" or "gemini".

    Returns:
        str: The model endpoint URL or model name.
    """
    if provider == "lmstudio":
        # For LM Studio, the base URL is the primary "endpoint" for the API.
        # The actual model might be selected on the server or passed in payload.
        return LMSTUDIO_API_BASE_URL
    # elif provider == "gemini_pro":
    #     return GEMINI_PRO_MODEL_NAME # Requires GEMINI_PRO_MODEL_NAME to be defined
    # elif provider == "gemini_flash":
    #     return GEMINI_FLASH_MODEL_NAME # Requires GEMINI_FLASH_MODEL_NAME to be defined
    else:
        # Fallback or default behavior
        logger.warning(f"Unknown LLM provider '{provider}' in get_llm_model_endpoint. Defaulting to LMStudio URL.")
        return LMSTUDIO_API_BASE_URL

# Example of how other modules might get the LM Studio URL:
# from config import settings
# lm_studio_url = settings.LMSTUDIO_API_BASE_URL
# or
# lm_studio_url = settings.get_llm_model_endpoint("lmstudio")

# Setup a basic logger for warnings within this settings file itself if needed.
# This helps if, for example, environment variables are missing during initial load.
if __name__ != '__main__': # Avoid running this if the script is executed directly
    import logging as settings_logging
    logger = settings_logging.getLogger(__name__)
    # Example: Check if LMSTUDIO_API_BASE_URL was defaulted
    if LMSTUDIO_API_BASE_URL == "http://localhost:1234/v1" and not os.getenv("LMSTUDIO_API_BASE_URL"):
        logger.info("LMSTUDIO_API_BASE_URL is using the default value 'http://localhost:1234/v1'. "
                    "Set LMSTUDIO_API_BASE_URL in your .env file to override.")

