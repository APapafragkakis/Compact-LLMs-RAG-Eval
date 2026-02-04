import ollama

class Generator:
    def __init__(self, config):
        self.config = config
        self.model_name = config["model"]
        self.temperature = config["temperature"]
        self.system_message = config["prompts"].get("system", None)

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt}
        ]
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_ctx": 8192,
                "num_predict": 500
            }
        )
        return response['message']['content']