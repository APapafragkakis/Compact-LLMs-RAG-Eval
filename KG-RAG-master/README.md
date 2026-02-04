# Installation
1. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```
2. Download and install an embeddings model e.g. [nomic-embed-text](https://ollama.com/library/nomic-embed-text)


3. Download and install an LLM e.g. [LLama3.2](https://ollama.com/library/llama3.2)

4. Modify the [config.yaml](app/chatbot//config.yaml) if needed

# Run examples
python3 -m unittest ./examples/example.py 