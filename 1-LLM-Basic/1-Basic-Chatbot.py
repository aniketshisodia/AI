import ollama 

response = ollama.chat(
    model = "gemma3",
    messages = [
        {
            "role" : "user",
            "content" : "What makes coffe taste good?"
        }
    ]
)

answer = response['message']['content']

print(f"Answer: {answer}")