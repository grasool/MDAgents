from utils import determine_difficulty, process_basic_query, process_advanced_query

def test_integration():
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

    difficulty = determine_difficulty(question, "adaptive")
    print("Determined Difficulty:", difficulty)

    if difficulty == "basic":
        result = process_basic_query(question, examplers, model_name, args)
    else:
        result = process_advanced_query(question, examplers, model_name, args)

    print("Final Decision:", result)

if __name__ == "__main__":
    test_integration()
