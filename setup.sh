#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"
PIP_TRUSTED=( --trusted-host pypi.org --trusted-host files.pythonhosted.org )

python -m pip install -U pip wheel setuptools -i "$PYPI_INDEX" "${PIP_TRUSTED[@]}"
pip install -e ".[dev]" -i "$PYPI_INDEX" "${PIP_TRUSTED[@]}"

python main.py build-graph
python main.py index

echo ""
echo "Сборка графа и индекс Qdrant для графа готовы."
echo "Индекс документов (опционально): python main.py llm-index  или  python -m vernadskiy_boltalka.rag_pipeline.embeddings --recreate"
echo "Чат: ./run.sh"
