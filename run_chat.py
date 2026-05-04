# run_chat.py
from model_utils import generate_answer


def main():
    print("=" * 60)
    print("КОНСОЛЬНЫЙ ЧАТ С ДООБУЧЕННОЙ МОДЕЛЬЮ")
    print("Введите 'выход' для завершения")
    print("=" * 60)

    history = []

    while True:
        user_message = input("\nВы: ").strip()

        if user_message.lower() in ["выход", "exit", "quit"]:
            print("Завершение работы.")
            break

        if not user_message:
            continue

        result = generate_answer(user_message, history)
        answer = result["response"]

        print(f"\nМодель: {answer}")
        print(f"[latency: {result['latency']:.2f} сек.; throughput: {result['throughput']:.2f} ток/сек]")

        history.append({"user": user_message, "bot": answer})


if __name__ == "__main__":
    main()
