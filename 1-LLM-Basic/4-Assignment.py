import ollama

system_prompt = {
    "role": "system",
    "content": "You are a friendly pizza shop assistant. Keep answers short and mention your love for pizza."
}

conversation = [system_prompt]

questions = [
    "What toppings do you have?",
    "Do you have gluten-free options?",  # Added space and question mark
    "What's your special today?"  # Added "today" for specificity
]

for question in questions:
    conversation.append({"role": "user", "content": question})
    
    response = ollama.chat(
        model="gemma3",  # or "gemma3:2b" if you want faster responses
        messages=conversation,
        options={"temperature": 0.5}  # Optional: control creativity
    )
    
    print(f"\n🍕 Customer: {question}")
    print(f"🤖 AI: {response['message']['content']}")
    
    conversation.append({
        "role": "assistant", 
        "content": response['message']['content']
    })

print("\n✅ Conversation complete!")