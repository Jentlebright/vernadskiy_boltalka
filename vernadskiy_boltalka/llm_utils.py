from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from vernadskiy_boltalka.config import config


def _qwen_model_name() -> str:
    return config.QWEN_MODEL or "qwen2.5"


def get_chat_llm():
    if config.OLLAMA_MODEL:
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
        )
    if config.USE_QWEN:
        return ChatOpenAI(
            base_url=config.QWEN_RUADAPT_BASE_URL,
            api_key=config.QWEN_RUADAPT_API_KEY,
            model=_qwen_model_name(),
        )
    if config.USE_VSEGPT:
        return ChatOpenAI(
            base_url=config.VSEGPT_API_URL,
            api_key=config.VSEGPT_API_KEY,
            model=config.VSEGPT_MODEL,
        )
    if config.USE_BOTHUB:
        return ChatOpenAI(
            base_url=config.BOTHUB_BASE_URL,
            api_key=config.BOTHUB_API_KEY,
            model=config.BOTHUB_MODEL,
        )
    if config.OPENAI_API_KEY.strip():
        return ChatOpenAI(
            model=config.MODEL or "gpt-4o-mini",
            api_key=config.OPENAI_API_KEY,
        )
    raise RuntimeError(
        "Не настроена модель. Укажи в `.env`: OPENAI_API_KEY (+ MODEL), "
        "или USE_BOTHUB+BOTHUB_*, или USE_VSEGPT+VSEGPT_*, или OLLAMA_MODEL, или USE_QWEN+QWEN_*."
    )
