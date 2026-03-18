import os
from pathlib import Path

_env_path = Path(__file__).parent / "_env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.split("#")[0].strip().strip("'\"")
                os.environ[k] = v


# МОИ ОБХОДЫ ПРОКСИ, потом убрать
_noproxy = os.environ.get("NO_PROXY", "")
_add = ",".join(h for h in ("localhost", "127.0.0.1") if h not in _noproxy)
if _add:
    os.environ["NO_PROXY"] = f"{_add},{_noproxy}".strip(",")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
BOTHUB_BASE_URL = os.getenv("BOTHUB_BASE_URL", "")
BOTHUB_API_KEY = os.getenv("BOTHUB_API_KEY", "")
BOTHUB_MODEL = os.getenv("BOTHUB_MODEL_GASPROM", "gemini-2.0-flash-lite-001")
USE_BOTHUB = bool(BOTHUB_BASE_URL and BOTHUB_API_KEY)
VSEGPT_API_URL = os.getenv("VSEGPT_API_URL", "")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY", "")
VSEGPT_MODEL = os.getenv("VSEGPT_MODEL", "meta-llama/llama-3-8b-instruct")
USE_VSEGPT = bool(VSEGPT_API_URL and VSEGPT_API_KEY)
QWEN_RUADAPT_BASE_URL = os.getenv("QWEN_RUADAPT_BASE_URL", "")
QWEN_RUADAPT_API_KEY = os.getenv("QWEN_RUADAPT_API_KEY", "")
USE_QWEN = bool(QWEN_RUADAPT_BASE_URL and QWEN_RUADAPT_API_KEY)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")
QDRANT_PATH = os.getenv("QDRANT_PATH", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = "vernadsky_rag"
