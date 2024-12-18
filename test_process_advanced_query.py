from utils import process_advanced_query

def test_process_advanced_query():
    question = "What is the treatment for Type 2 Diabetes?"
    examplers = [
        {
            "question": "Which medication is first-line therapy for Type 2 Diabetes?",
            "options": {"A": "Metformin", "B": "Insulin", "C": "Sulfonylureas"},
            "answer_idx": "A",
            "answer": "Metformin",
        }
    ]
    class Args:
        dataset = "medqa"
    args = Args()
    model_name = "llama3.1"

    try:
        result = process_advanced_query(question, examplers, model_name, args)
        print("Advanced Query Test Result:", result)
    except Exception as e:
        print(f"Error in test_process_advanced_query: {e}")

if __name__ == "__main__":
    test_process_advanced_query()
