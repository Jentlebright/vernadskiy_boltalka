import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI

import config

DATA_DIR = Path(__file__).parent / "vernadskiy_data"
GRAPH_PATH = Path(__file__).parent / "data" / "vernadsky_graph.json"
CHUNK_SIZE = 2500
CHUNK_OVERLAP = 200

EXTRACT_PROMPT = """Извлеки из текста В.И. Вернадского следующие элементы.

ТИПЫ УЗЛОВ (nodes):
- concept: научные понятия (ноосфера, биосфера, живое вещество, научная мысль и т.п.)
- emotion: эмоции, чувства (вдохновение, одобрение, тревога и т.п.)
- value: ценности (важность науки, единство природы, гуманизм и т.п.)

ТИПЫ СВЯЗЕЙ (edges):
- source: id узла-источника
- target: id узла-цели
- relation: тип связи (вызывает, предшествует, ценностное отношение и т.п.)
- quote: краткая цитата Вернадского (до 200 символов), иллюстрирующая связь
- intensity: необязательно (низкая, средняя, высокая)
- level: необязательно, для ценностных отношений (одобряет, осуждает и т.п.)

ФОРМАТ ОТВЕТА — валидный JSON:
{{"nodes": [{{"id": "строка", "type": "concept|emotion|value"}}, ...],
 "edges": [{{"source": "...", "target": "...", "relation": "...", "quote": "..."}}, ...]}}

Извлекай только то, что явно выражено в тексте. Не выдумывай.
Если ничего подходящего нет — верни {{"nodes": [], "edges": []}}

ТЕКСТ:
---
{text}
---
JSON:"""


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        print(f"  PDF {path.name}: ошибка — {e}")
        return ""


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"  DOCX {path.name}: ошибка — {e}")
        return ""


def _extract_epub(path: Path) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(path)
        parts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(item.get_content(), "html.parser")
                parts.append(soup.get_text(separator="\n"))
        return "\n\n".join(parts)
    except Exception as e:
        print(f"  EPUB {path.name}: ошибка — {e}")
        return ""


def load_texts_from_dir(data_dir: Path) -> list[tuple[str, str]]:
    results = []
    if not data_dir.exists():
        print(f"Папка не найдена: {data_dir}")
        return results

    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() == ".pdf":
            t = _extract_pdf(f)
        elif f.suffix.lower() == ".docx":
            t = _extract_docx(f)
        elif f.suffix.lower() == ".epub":
            t = _extract_epub(f)
        else:
            continue
        text = (t or "").strip()
        if text:
            results.append((f.name, text))
            print(f"  Загружен: {f.name} ({len(text)} символов)")

    return results


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if end < len(text):
            last_space = chunk.rfind(" ", size - overlap)
            if last_space > size // 2:
                end = start + last_space + 1
                chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end
    return [c for c in chunks if c]


def _get_llm():
    if config.OLLAMA_MODEL:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=config.OLLAMA_BASE_URL,
            model=config.OLLAMA_MODEL,
        )
    if config.USE_QWEN:
        from chat_graph import _get_qwen_model
        return ChatOpenAI(
            base_url=config.QWEN_RUADAPT_BASE_URL,
            api_key=config.QWEN_RUADAPT_API_KEY,
            model=_get_qwen_model(),
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
    return ChatOpenAI(
        model=config.MODEL or "gpt-4o-mini",
        api_key=config.OPENAI_API_KEY,
    )


def _parse_json_from_response(response: str) -> dict | None:
    text = response.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def extract_from_chunk(llm, chunk: str) -> tuple[list, list]:
    prompt = EXTRACT_PROMPT.format(text=chunk[:3000])
    try:
        out = llm.invoke(prompt)
        content = out.content if hasattr(out, "content") else str(out)
        data = _parse_json_from_response(content)
        if data:
            nodes = data.get("nodes", [])
            edges = data.get("edges", [])
            return (nodes, edges)
    except Exception as e:
        print(f"  LLM ошибка: {e}")
    return ([], [])


def merge_graphs(all_nodes: list[dict], all_edges: list[dict]) -> dict:
    seen_nodes = {}
    for n in all_nodes:
        raw_id = (n.get("id") or "").strip()
        nid = raw_id.lower()
        if not nid:
            continue
        if nid not in seen_nodes:
            seen_nodes[nid] = {"id": raw_id, "type": n.get("type", "concept")}

    nodes = list(seen_nodes.values())
    id_to_canonical = {n["id"].lower(): n["id"] for n in nodes}

    edges = []
    seen_edges = set()
    for e in all_edges:
        s_raw = (e.get("source") or "").strip()
        t_raw = (e.get("target") or "").strip()
        s = s_raw.lower()
        t = t_raw.lower()
        q = (e.get("quote") or "").strip()[:300]
        if not s or not t:
            continue
        source = id_to_canonical.get(s, s_raw)
        target = id_to_canonical.get(t, t_raw)
        key = (s, t, q[:50])
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edge = {
            "source": source,
            "target": target,
            "relation": (e.get("relation") or "").strip() or "связь",
            "quote": q,
        }
        if e.get("intensity"):
            edge["intensity"] = e.get("intensity")
        if e.get("level"):
            edge["level"] = e.get("level")
        edges.append(edge)

    return {"nodes": nodes, "edges": edges}


def build_graph_from_data(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or DATA_DIR
    texts = load_texts_from_dir(data_dir)
    if not texts:
        print("Нет текстов для обработки.")
        return {"nodes": [], "edges": []}

    llm = _get_llm()
    all_nodes = []
    all_edges = []

    for fname, text in texts:
        chunks = chunk_text(text)
        print(f"\n{fname}: {len(chunks)} чанков")
        for i, chunk in enumerate(chunks):
            n, e = extract_from_chunk(llm, chunk)
            all_nodes.extend(n)
            all_edges.extend(e)
            if (i + 1) % 3 == 0:
                print(f"  обработано {i + 1}/{len(chunks)}")

    graph = merge_graphs(all_nodes, all_edges)
    print(f"\nИтого: {len(graph['nodes'])} узлов, {len(graph['edges'])} связей")
    return graph


def run():
    print(f"Источник: {DATA_DIR}")
    graph = build_graph_from_data()
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Граф сохранён: {GRAPH_PATH}")


if __name__ == "__main__":
    run()
