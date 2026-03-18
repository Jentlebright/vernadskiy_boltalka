import config
from langchain_core.messages import AIMessage, HumanMessage

from chat_graph import chat
from graph_store import build_index


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "index":
        build_index()
        print("Индекс графа знаний построен.")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build-graph":
        from build_graph import run as build_run
        build_run()
        return
    history = []
    print("Болталка с Вернадским. Режимы: научные темы (ноосфера, биосфера...) / личное.")
    print("Команды: /выход, /режим")
    while True:
        try:
            inp = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания.")
            break
        if not inp:
            continue
        if inp.lower() in ("/выход", "/exit", "/quit"):
            print("До свидания.")
            break
        if inp.lower() == "/режим":
            print("Режим выбирается автоматически по теме сообщения.")
            continue
        try:
            reply = chat(inp, history[-10:] if len(history) > 10 else history)
            history.append(HumanMessage(content=inp))
            history.append(AIMessage(content=reply))
            if len(history) > 20:
                history = history[-20:]
            print(f"Вернадский: {reply}")
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
