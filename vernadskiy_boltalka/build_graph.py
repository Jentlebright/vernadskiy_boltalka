import json
import os
import re
import time

from vernadskiy_boltalka.llm_utils import get_chat_llm
from vernadskiy_boltalka.paths import project_root

_ROOT = project_root()
DATA_DIR = os.path.join(_ROOT, "vernadskiy_data")
GRAPH_PATH = os.path.join(_ROOT, "data", "vernadsky_graph.json")
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


def load_texts_from_dir(data_dir: str) -> list[tuple[str, str]]:
    from vernadskiy_boltalka.rag_pipeline.preprocessing_data import Preprocessor
    return Preprocessor().load_from_dir(data_dir)


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


def _log_slice_health(raw_len: int, chunks: list[str]) -> None:
    if not chunks:
        print(f"  чтение: {raw_len} симв. → чанков 0 (проверь PDF/извлечение текста)", flush=True)
        return
    lens = [len(c) for c in chunks]
    mn, mx = min(lens), max(lens)
    avg = sum(lens) // len(lens)
    short = sum(1 for x in lens if x < 120)
    hint = "ок" if short <= max(1, len(lens) // 5) else f"много коротких ({short}/{len(lens)})"
    print(
        f"  чтение: {raw_len:,} симв. → {len(chunks)} чанков | длина симв. min–ср–max: {mn}–{avg}–{mx} | {hint}",
        flush=True,
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


def build_graph_from_data(data_dir: str | None = None) -> dict:
    data_dir = data_dir or DATA_DIR
    texts = load_texts_from_dir(data_dir)
    if not texts:
        print("Нет текстов для обработки.", flush=True)
        return {"nodes": [], "edges": []}

    llm = get_chat_llm()
    all_nodes = []
    all_edges = []

    for fname, text in texts:
        chunks = chunk_text(text)
        print(f"\n{fname}", flush=True)
        _log_slice_health(len(text), chunks)
        nreq = len(chunks)
        print(
            f"  LLM: будет {nreq} запросов к модели; между строками может быть тишина 30 с–несколько мин — это ожидание API.",
            flush=True,
        )
        for i, chunk in enumerate(chunks):
            t0 = time.perf_counter()
            n, e = extract_from_chunk(llm, chunk)
            dt = time.perf_counter() - t0
            all_nodes.extend(n)
            all_edges.extend(e)
            print(
                f"  LLM {i + 1}/{nreq}  {dt:5.1f}s  +{len(n)} узл. +{len(e)} рёбер",
                flush=True,
            )

    graph = merge_graphs(all_nodes, all_edges)
    print(f"\nИтого: {len(graph['nodes'])} узлов, {len(graph['edges'])} связей", flush=True)
    return graph


def run():
    print(f"Источник: {DATA_DIR}", flush=True)
    graph = build_graph_from_data()
    graph_dir = os.path.dirname(GRAPH_PATH)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"Граф сохранён: {GRAPH_PATH}", flush=True)


if __name__ == "__main__":
    run()
