import sys
sys.path.insert(0, '.')
from src.text_pipeline import run_text_pipeline
from src.logger import print_log_summary

print("\n" + "="*60)
print("  SMART CAMPUS CHATBOT - Text Pipeline")
print("  Type your question and press Enter")
print("  Type 'logs' to see conversation history")
print("  Type 'exit' to quit")
print("="*60 + "\n")

while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not query:
        continue

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if query.lower() == "logs":
        print_log_summary()
        continue

    result = run_text_pipeline(query)

    print(f"\nBot: {result['response']}")
    print(f"     [intent={result['intent']} | conf={result['intent_confidence']:.2f} | matched={result['matched_location']} | llm={'yes' if result['llm_used'] else 'no'}]")
    print()
