# lesson2_system_prompt.py
import ollama

# Define the AI's role
system_message = {
    "role": "system",
    "content": """You are a friendly barista at Brew Haven Cafe. 
    You're knowledgeable about coffee, tea, and pastries. 
    Keep responses short and welcoming."""
}

messages = [system_message]

while True:
    user_input = input("\nYou: ")
    if user_input == 'quit':
        break
    
    messages.append({"role": "user", "content": user_input})
    
    response = ollama.chat(
        model="gemma3",
        messages=messages,
        options={"temperature": 0.7}
    )
    
    ai_reply = response['message']['content']
    messages.append({"role": "assistant", "content": ai_reply})
    
    print(f"\n☕ {ai_reply}")