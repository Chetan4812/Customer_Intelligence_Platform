import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from services.rag.rag_engine import RAGEngine

def run_eval():
    engine = RAGEngine.get()
    
    questions = [
        ("What are the most common billing issues?", True),
        ("Why was I charged twice for my subscription?", True),
        ("Did the chat service go down recently?", True),
        ("Who won the Superbowl last year?", False),
        ("What is the capital of France?", False)
    ]
    
    passed = 0
    for q, should_answer in questions:
        print(f"\nQ: {q}")
        res = engine.answer(q, top_k=3)
        if should_answer and len(res.sources) > 0:
            print("[PASS] Successfully answered domain question.")
            passed += 1
        elif not should_answer and len(res.sources) == 0:
            print("[PASS] Successfully refused out-of-domain question.")
            passed += 1
        else:
            print("[FAIL] Unexpected behavior.")
            print(f"Sources found: {len(res.sources)}")
            
    print(f"\nEvaluation complete: {passed}/{len(questions)} passed.")
    if passed != len(questions):
        sys.exit(1)

if __name__ == "__main__":
    run_eval()
