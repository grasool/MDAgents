import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
# import google.generativeai as genai
# from openai import OpenAI
from pptree import *

import ollama
from ollama import chat, ChatResponse

class Agent:
    def __init__(self, instruction, role, model_name='llama3.1'):
        """
        Initialize the agent with a system-level instruction, role, and model name.

        Args:
            instruction (str): The system-level instruction for the agent (e.g., "You are a medical expert.").
            role (str): The role of the agent (e.g., "medical expert", "recruiter").
            model_name (str): The name of the model to use with Ollama (default: "llama3.1").
        """
        self.instruction = instruction  # System-level instruction for the agent
        self.role = role  # Role of the agent
        self.model_name = model_name  # Name of the model to use
        self.history = [{"role": "system", "content": instruction}]  # Initialize conversation history

    def chat(self, message):
        """
        Interact with the Ollama model by sending a message and receiving a response.

        Args:
            message (str): The user's input message to the agent.

        Returns:
            str: The assistant's response.
        """
        # Add the user's message to the conversation history
        self.history.append({"role": "user", "content": message})

        # Call the Ollama API to get the assistant's response
        response: ChatResponse = chat(
            model=self.model_name,
            messages=self.history
        )

        # Extract the assistant's response
        assistant_message = response.message.content

        # Add the assistant's response to the conversation history
        self.history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def temp_responses(self, message, temperatures=[0.0, 0.5, 1.0]):
        """
        Generate responses with different temperatures (levels of randomness).

        Args:
            message (str): The user's input message.
            temperatures (list of float): List of temperature values for generating diverse responses.

        Returns:
            dict: A dictionary where keys are temperature values and values are responses.
        """
        responses = {}

        for temperature in temperatures:
            # Create a temporary message history to avoid modifying the main history
            temp_history = self.history + [{"role": "user", "content": message}]

            # Call Ollama's chat API with the specified temperature
            try:
                response: ChatResponse = chat(
                    model=self.model_name,
                    messages=temp_history,
                    options={"temperature": temperature}
                )
                # Extract the response content
                responses[temperature] = response.message.content
            except Exception as e:
                # Handle errors gracefully
                responses[temperature] = f"Error: {e}"

        return responses

class Group:
    def __init__(self, goal, members, question, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_name='llama3.1')
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            try:
                delivery = lead_member.chat(delivery_prompt)
            except:
                delivery = assist_members[0].chat(delivery_prompt)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)

            return response

        elif comm_type == 'external':
            return

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        raise ValueError(f"Unsupported model: {model_name}")
        # genai.configure(api_key=os.environ['genai_api_key'])
        # return genai, None
    elif 'gpt' in model_name:
        raise ValueError(f"Unsupported model: {model_name}")
        # client = OpenAI(api_key=os.environ['openai_api_key'])
        # return None, client
    elif 'ollama' in model_name:
        # Initialize Ollama (it runs locally, so no API keys are needed)
        return ollama, None
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = r'D:\Works\SomeCodes\MDAgents\data\medqa\test.jsonl'  #../data/{dataset}/test.jsonl'
    print(dataset)
    print(test_path)
    with open(test_path, 'r', encoding='utf-8') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = r'D:\Works\SomeCodes\MDAgents\data\medqa\train.jsonl' #f'../data/{dataset}/train.jsonl'
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty_mode):
    """
    Determine the difficulty level of a given question.
    """
    if difficulty_mode == 'adaptive':
        # Use the Agent class to decide the difficulty of the question
        medical_agent = Agent(
            instruction='''You are a medical expert whose job is to classify the difficulty level of medical questions.\
              Respond with one of the following: "basic", "intermediate", or "advanced". Provide no additional explanations..''',
            role='medical expert',
            model_name='llama3.1'  # Specify the model name for Ollama
        )

        response = medical_agent.chat(f"Question: {question}")
        if 'basic' in response.lower():
            return 'basic'
        elif 'intermediate' in response.lower():
            return 'intermediate'
        elif 'advanced' in response.lower():
            return 'advanced'
        else:
            raise ValueError(f"Unexpected response from Agent: {response}")
    else:
        # Default difficulty mode
        return difficulty_mode


def process_basic_query(question, examplers, model, args):
    """
    Processes a basic medical knowledge query by generating a final decision using an agent.

    Args:
        question (str): The question to process.
        examplers (list): Example cases for the agent to reference.
        model (str): The name of the model (e.g., 'llama3.1').
        args (object): Arguments containing dataset information.

    Returns:
        dict: The final decision made by the agent.
    """
    # Initialize the medical agent
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_name=model)

    # Create examples for the agent
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"

            # Ask the agent to generate reasoning for the example
            exampler_reason = medical_agent.chat(
                f"""You are a helpful medical agent. Below is an example of a medical knowledge question and answer. 
                After reviewing the question and answer, provide 1-2 sentences of reasoning.\n\n
                Question: {exampler_question}\n\nAnswer: {exampler_answer}"""
            )

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)

    # Initialize the single agent to answer the question
    single_agent = Agent(
        instruction='You are a helpful assistant that answers multiple-choice questions about medical knowledge.',
        role='medical expert',
        model_name=model
    )
    single_agent.chat('You are a helpful assistant that answers multiple-choice questions about medical knowledge.')

    # Generate the final decision
    final_decision = single_agent.temp_responses(
        f"""The following are multiple-choice questions (with answers) about medical knowledge. 
        Let's think step by step.\n\n**Question:** {question}\nAnswer: """
    )
    
    return final_decision

def process_advanced_query(question, examplers, model_name, args):
    """
    Process a medical query requiring collaboration between multiple agents.

    Args:
        question (str): The medical query to process.
        examplers (list): Example questions and answers for reference.
        model_name (str): The name of the model to use with the Agent class.
        args (Namespace): Additional arguments, such as dataset information.

    Returns:
        dict: The final decision for the query.
    """
    cprint("[INFO] Step 1. Expert Recruitment", "yellow", attrs=["blink"])

    # Step 1: Recruit Experts
    recruit_prompt = (
        "You are an experienced medical expert who recruits a group of experts with diverse "
        "identities to discuss and solve the given medical query."
    )
    recruiter = Agent(instruction=recruit_prompt, role="recruiter", model_name=model_name)
    recruiter.chat(recruit_prompt)

    # Recruit experts based on the question
    num_agents = 5  # Adjust the number of experts as needed
    recruitment_response = recruiter.chat(
    f"Question: {question}\n"
    f"You can recruit {num_agents} experts in different medical expertise. "
    f"Considering the medical question and options, return a recruitment plan in the following format:\n"
    f"Role 1 - Description 1\n"
    f"Role 2 - Description 2\n"
    f"...\n\n"
    f"Make sure each role and description are separated by ' - '. Do not include any additional information."
)


    # Parse recruited experts
    agents_data = [line.strip() for line in recruitment_response.split("\n") if line.strip()]
    agents = []

    for agent_data in agents_data:
        print("Agents Data:", agent_data)

        if " - " in agent_data:
            # Safely split the role and description
            role, description = agent_data.split(" - ", 1)
            role = role.strip()
            description = description.strip()

            # Initialize and set up the Agent
            agent = Agent(
                instruction=f"You are a {role} who {description}.",
                role=role,
                model_name=model_name,
            )
            agent.chat(f"You are a {role} who {description}.")
            agents.append(agent)
        else:
            # Handle unexpected format gracefully
            print(f"Invalid format for agent data: {agent_data}")
            agents.append(
                Agent(
                    instruction="You are an undefined role with missing description.",
                    role="undefined",
                    model_name=model_name,
                )
            )


    # Step 2: Collaborative Decision-Making
    cprint("[INFO] Step 2. Collaborative Decision-Making", "yellow", attrs=["blink"])

    # Few-shot examples for context
    fewshot_examplers = ""
    if args.dataset == "medqa":
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            question = f"[Example {ie + 1}]\n{exampler['question']}"
            options = [f"({k}) {v}" for k, v in exampler["options"].items()]
            random.shuffle(options)
            question += " " + " ".join(options)
            answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            reason = recruiter.chat(
                f"Below is a medical question and answer. Can you provide 1-2 sentences of reasoning?\n\n"
                f"Question: {question}\n\nAnswer: {answer}"
            )
            fewshot_examplers += f"{question}\n{answer}{reason}\n\n"

    # Get each agent's initial opinion
    agent_opinions = {}
    for agent in agents:
        print('Getting opinion from experts:', agent.role)
        opinion = agent.chat(
            f"Given the examples, please return your answer to the medical query among the options provided.\n\n"
            f"{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like: Answer: (A) Explanation."
        )
        agent_opinions[agent.role] = opinion
        print(f"{agent.role}: {opinion}")

    # Summarize and finalize decision
    summarizer = Agent(
        instruction="You are a medical assistant who summarizes and synthesizes input from multiple experts.",
        role="summarizer",
        model_name=model_name,
    )
    summarizer.chat(
        "You are a medical assistant who summarizes and synthesizes input from multiple experts."
    )

    summary = summarizer.chat(
        "Here are the opinions from different medical experts:\n\n"
        + "\n".join(f"{role}: {opinion}" for role, opinion in agent_opinions.items())
        + "\n\nPlease synthesize and provide a final decision."
    )
    final_summary = summary
    print(f"Final Summary: {final_summary}")
    return {"final_decision": summary}


