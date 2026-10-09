import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DASHSCOPE_API_KEY = os.getenv("key")
QWEN_BASE_URL = os.getenv("baseurl")
QWEN_MODEL = os.getenv("QWEN_MODEL")

if not DASHSCOPE_API_KEY:
    raise RuntimeError("DASHSCOPE_API_KEY is not configured")