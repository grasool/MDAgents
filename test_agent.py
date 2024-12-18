#test_agent

from utils import Agent

def test_agent():
    # Define an instruction and role for the agent
    instruction = "You are a helpful assistant who answers medical questions."
    role = "medical expert"

    # Initialize the agent
    agent = Agent(instruction=instruction, role=role, model_name="llama3.1")

    # Test the agent with a sample question
    question = "What are the symptoms of hypertension?"
    response = agent.chat(question)

    # Print the results
    print("User Question:", question)
    print("Agent Response:", response)

# Run the test
if __name__ == "__main__":
    test_agent()
