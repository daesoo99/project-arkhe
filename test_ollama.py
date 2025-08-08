import ollama

def test_ollama_connection():
    """
    Tests the connection to the Ollama server and a local model.
    """
    try:
        # Ensure you have pulled a model, e.g., 'ollama pull llama3'
        response = ollama.chat(
            model='llama3', 
            messages=[{'role': 'user', 'content': '1+1=?'}]
        )
        print("Ollama connection successful!")
        print(f"Response from llama3: {response['message']['content']}")
    except Exception as e:
        print(f"An error occurred while testing Ollama: {e}")
        print("Please ensure the Ollama application is running and you have pulled the 'llama3' model.")

if __name__ == "__main__":
    test_ollama_connection()
