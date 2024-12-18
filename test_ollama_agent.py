from ollama import chat, ChatResponse

class Agent:
    def __init__(self, instruction, role, model_name='llama3.2'):
        """
        Initialize the agent with an instruction, role, and model name.
        """
        self.instruction = instruction  # System-level prompt
        self.role = role  # Agent's role (e.g., "medical expert")
        self.model_name = model_name  # Name of the LLaMA model to use with Ollama
        self.history = [{"role": "system", "content": instruction}]  # Conversation history

    def chat(self, message):
        """
        Send a message to the Ollama model and return its response.
        """
        # Append the user's message to the conversation history
        self.history.append({"role": "user", "content": message})

        # Use the Ollama `chat` function to interact with the model
        response: ChatResponse = chat(
            model=self.model_name,
            messages=self.history
        )

        # Extract the assistant's response
        assistant_message = response.message.content
        # Add the assistant's response to the conversation history
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message


# Test the Agent class
def main():
    # Define an instruction and role for the agent
    instruction = "You are a helpful medical expert who answers questions about medical knowledge."
    role = "medical expert"

    # Create the Agent instance
    agent = Agent(instruction=instruction, role=role, model_name="llama3.2")

    # Test a question
    test_question = "What are the symptoms of hypertension?"
    print(f"User: {test_question}")

    # Get the response
    response = agent.chat(test_question)
    print(f"Agent: {response}")


if __name__ == "__main__":
    main()
