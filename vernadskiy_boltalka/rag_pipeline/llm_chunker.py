import json
import re
from typing import Any

from pydantic import BaseModel, Field

from vernadskiy_boltalka.llm_utils import get_chat_llm


class _ChunkItem(BaseModel):
    text: str = ""


class _ChunkBundle(BaseModel):
    chunks: list[_ChunkItem]
    recommended_overlap: int = Field(default=96, ge=16, le=400)


def _parse_json_obj(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _rough_blocks(text: str, target: int) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text] if text.strip() else []
    blocks: list[str] = []
    buf: list[str] = []
    cur = 0
    for p in paras:
        plen = len(p)
        if buf and cur + plen > target:
            blocks.append("\n\n".join(buf))
            buf = [p]
            cur = plen
        else:
            buf.append(p)
            cur += plen + 2
    if buf:
        blocks.append("\n\n".join(buf))
    return blocks or ([text] if text.strip() else [])


def _invoke_bundle(llm: Any, block: str) -> _ChunkBundle | None:
    prompt = (
        "Разбей текст на смысловые фрагменты для семантического поиска по корпусу Вернадского. "
        "Каждый фрагмент — связная мысль; сохраняй терминологию автора. "
        "Оцени recommended_overlap: разумный хвост перекрытия между соседними фрагментами в символах.\n"
        "Ответ строго JSON: {\"chunks\":[{\"text\":\"...\"}],\"recommended_overlap\":число}\n\n"
        f"Текст:\n---\n{block}\n---"
    )
    try:
        structured = getattr(llm, "with_structured_output", None)
        if structured:
            try:
                chain = structured(_ChunkBundle)
                return chain.invoke(prompt)
            except Exception:
                pass
        out = llm.invoke(prompt)
        content = out.content if hasattr(out, "content") else str(out)
        data = _parse_json_obj(content)
        if not data:
            return None
        return _ChunkBundle.model_validate(data)
    except Exception:
        return None


def chunk_text_with_llm(text: str) -> tuple[list[str], int]:
    text = (text or "").strip()
    if not text:
        return [], 80
    llm = get_chat_llm()
    blocks = _rough_blocks(text, 14000)
    texts: list[str] = []
    overlaps: list[int] = []
    for block in blocks:
        bundle = _invoke_bundle(llm, block)
        if not bundle or not bundle.chunks:
            continue
        for c in bundle.chunks:
            t = (c.text or "").strip()
            if t:
                texts.append(t)
        overlaps.append(bundle.recommended_overlap)
    if not texts:
        return [text], 80
    overlap = int(sum(overlaps) / len(overlaps)) if overlaps else 96
    return texts, max(32, min(overlap, 320))


def chunk_corpus_with_llm(pairs: list[tuple[str, str]]) -> tuple[list[dict], int]:
    out: list[dict] = []
    overs: list[int] = []
    for source, body in pairs:
        parts, ov = chunk_text_with_llm(body)
        overs.append(ov)
        for i, part in enumerate(parts):
            out.append({"text": part, "meta": {"source": source, "chunk_idx": i}})
    agg = int(sum(overs) / len(overs)) if overs else 96
    return out, max(32, min(agg, 320))
