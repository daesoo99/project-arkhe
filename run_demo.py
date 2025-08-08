import ollama
from dotenv import load_dotenv
import os

load_dotenv()

def run_demo():
    """
    This function runs a demo of the Ollama API.
    """
    try:
        response = ollama.chat(model='gemma:2b', messages=[
            {
                'role': 'user',
                'content': 'Why is the sky blue?',
            },
        ])
        print(response['message']['content'])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_demo()
