# ОБЩАЯ ИДЕЯ
1. пакет vernadskiy_boltalka/ - весь код приложения (чат, граф, graph_store, rag_pipeline)
2. paths.project_root() - корень репозитория (рядом лежат data/, vernadskiy_data/), чтобы не завязываться на __file__ в корне
3. pyproject.toml - зависимости, метаданные, console script vernadskiy-chat.
tests/ — pytest
4. файлы в корне (main.py, config.py, build_graph.py, …) - старые команды вроде python main.py и import config продолжают работать

## ЗАПУСК: ВАРИАНТ 1
```
pip install -e .

python main.py build-graph
python main.py index
python main.py
```
1. python main.py build-graph
    читает тексты из vernadskiy_data/, дергает LLM, вытаскивает узлы и рёбра графа, пишет data/vernadsky_graph.json;
    (тяжёлый офлайн-шаг: много запросов к модели, делается когда меняются книги или хочется пересобрать граф)

2. python main.py index
    берёт уже готовый vernadsky_graph.json, режет его на фрагменты для поиска, считает эмбеддинги и заливает в Qdrant;
    нужен для того, чтобы чат в режиме "диалог на специальные темы аавтора" мог доставать контекст из векторной базы

3. python main.py (без аргументов)
    запускает интерактивный чат в терминале — то, чем пользуются уже после того, как граф и индекс готовы

## ЗАПУСК: ВАРИАНТ 2
```
pip install -e .

python -m vernadskiy_boltalka
```

## ЗАПУСК: ВАРИАНТ 3
```
pip install -e .

vernadskiy-chat
```
    для этой команды существует cli.py (Command Line Interface), путь для которого прописан в pyptoject.toml [project.scripts]

## ЗАПУСК rag_pipeline
```
pip install -e .
```

1. только чанки: 
```python -m vernadskiy_boltalka.rag_pipeline.preprocessing_data```
фильтры: --data-dir, --out, --chunk-size, --chunk-overlap

2. чанки → эмбеддинги → удалённый Qdrant 
пересоздать коллекицю: 
```python -m vernadskiy_boltalka.rag_pipeline.embeddings --recreate```
по умолчанию читается vernadskiy_data/; если надо другой каталог, то 
```python -m vernadskiy_boltalka.rag_pipeline.embeddings --recreate --data-dir /путь/к/каталогу```
фильтры: --chunk-size, --chunk-overlap