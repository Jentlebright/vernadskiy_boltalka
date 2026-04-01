from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from vernadskiy_boltalka.config import config
from vernadskiy_boltalka.graph_store import get_retriever

SYSTEM_EXPERT = """Ты — языковая маска Владимира Ивановича Вернадского.
Отвечай от его лица, используя его стиль и терминологию.
Отвечай кратко, по существу. Используй контекст из базы знаний для подкрепления ответов.
Контекст из графа знаний:
{context}"""

SYSTEM_PERSONAL = """Ты — Владимир Иванович Вернадский в режиме неформального общения.
Отвечай от его лица на личные, бытовые темы.
Не углубляйся в науку, если не спрашивают. Будь тёплым и человечным."""


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mode: Literal["expert", "personal"]
    context: str


def _classify_mode(state: State) -> Literal["expert", "personal"]:
    msgs = state["messages"]
    if not msgs:
        return "personal"
    last = msgs[-1]
    if not isinstance(last, HumanMessage):
        return "personal"
    q = last.content.lower() if isinstance(last.content, str) else ""
    triggers = [
        "ноосфера", "биосфера", "наука", "живое вещество",
        "эволюция", "планета", "космос", "геология", "учёный",
        "вернадский", "твои идеи", "твои работы", "что такое",
    ]
    if any(t in q for t in triggers):
        return "expert"
    return "personal"


def expert_node(state: State) -> State:
    q = state["messages"][-1].content if state["messages"] else ""
    try:
        retriever = get_retriever(k=4)
        docs = retriever.invoke(str(q))
        context = "\n\n".join(d.page_content for d in docs)
    except Exception:
        context = ""
    return {"context": context, "mode": "expert"}


def personal_node(state: State) -> State:
    return {"context": "", "mode": "personal"}


def _get_qwen_model() -> str:
    return config.QWEN_MODEL or "qwen2.5"


def _get_llm():
    if config.USE_BOTHUB:
        return ChatOpenAI(
            base_url=config.BOTHUB_BASE_URL,
            api_key=config.BOTHUB_API_KEY,
            model=config.BOTHUB_MODEL,
        )

    raise RuntimeError(
        "Не настроена модель. Укажи один из вариантов в `.env` или `_env`: "
        "OPENAI_API_KEY (и при желании MODEL), "
        "или VSEGPT_API_URL+VSEGPT_API_KEY, "
        "или BOTHUB_BASE_URL+BOTHUB_API_KEY, "
        "или OLLAMA_MODEL."
    )


def generate_node(state: State) -> State:
    context = state.get("context", "")
    mode = state.get("mode", "personal")
    msgs = state["messages"]
    if mode == "expert":
        system = SYSTEM_EXPERT.format(context=context or "Нет дополнительного контекста.")
    else:
        system = SYSTEM_PERSONAL
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), MessagesPlaceholder(variable_name="messages")]
    )
    chain = prompt | llm
    response = chain.invoke({"messages": msgs})
    return {"messages": [response]}


def route_after_classify(state: State) -> str:
    mode = _classify_mode(state)
    if mode == "expert":
        return "expert"
    return "personal"


def build_graph() -> StateGraph:
    builder = StateGraph(State)
    builder.add_node("expert", expert_node)
    builder.add_node("personal", personal_node)
    builder.add_node("generate", generate_node)
    builder.add_conditional_edges("__start__", route_after_classify)
    builder.add_edge("expert", "generate")
    builder.add_edge("personal", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


def chat(message: str, history: list[BaseMessage] | None = None) -> str:
    graph = build_graph()
    msgs = list(history) if history else []
    msgs.append(HumanMessage(content=message))
    result = graph.invoke({
        "messages": msgs,
        "mode": "personal",
        "context": "",
    })
    out = result["messages"][-1]
    return out.content if hasattr(out, "content") else str(out)
