"""
Phase 1 Configuration Module
=============================
Loads all secrets from .env file using python-dotenv.
Centralises API keys, model names, and storage paths.
No key should be hardcoded anywhere else in the project.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root  (two levels up from this file)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → d:\Proj\PageIndex
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase1")

# ---------------------------------------------------------------------------
# NVIDIA NIM  (used for tree building / reasoning)
# ---------------------------------------------------------------------------
NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL: str = "meta/llama-3.1-70b-instruct"

# ---------------------------------------------------------------------------
# Groq  (used for fast querying)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# Storage paths  (relative to project root)
# ---------------------------------------------------------------------------
PDF_STORAGE_DIR: str = str(_PROJECT_ROOT / "multi_paper_rag" / "phase1" / "data" / "pdfs")
TREE_STORAGE_DIR: str = str(_PROJECT_ROOT / "multi_paper_rag" / "phase1" / "data" / "trees")

# Ensure storage directories exist on import
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
os.makedirs(TREE_STORAGE_DIR, exist_ok=True)


def validate_config() -> bool:
    """
    Validate that all required configuration values are present.

    Returns:
        True if configuration is valid.

    Raises:
        EnvironmentError: If any required API key is missing.
    """
    missing = []
    if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your_nvidia_nim_api_key_here":
        missing.append("NVIDIA_API_KEY")
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        missing.append("GROQ_API_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Please set them in your .env file at {_PROJECT_ROOT / '.env'}"
        )
    logger.info("Configuration validated successfully.")
    return True
