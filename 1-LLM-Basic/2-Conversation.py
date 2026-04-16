# ==========================================================
# LESSON 2: BUILDING A CONVERSATIONAL AI CHATBOT
# ==========================================================
#
# This program creates a simple terminal-based chatbot using
# the Ollama library and the Gemma 3 model.
#
# Key Concept:
# Unlike a one-time prompt, this chatbot remembers previous
# messages by storing conversation history in a list.
#
# Every time the user sends a new message:
# 1. We add that message to conversation history.
# 2. Send the ENTIRE conversation to the AI model.
# 3. AI generates response based on full context.
# 4. Store AI response back into history.
#
# This creates memory/context and makes the chatbot feel like
# a real conversation instead of isolated Q&A.
#
# Type 'quit' anytime to stop chatting.
# ==========================================================

import ollama


# Display chatbot welcome banner
print("=" * 50)
print("🤖 CHAT WITH AI (type 'quit' to exit)")
print("=" * 50)


# This list stores the complete conversation history
# Format Example:
# [
#   {"role": "user", "content": "Hi"},
#   {"role": "assistant", "content": "Hello!"}
# ]
conversation = []


# Infinite loop to keep chat running until user quits
while True:

    # Take input from user
    user_input = input("\nYou: ")


    # Check if user wants to end chat
    if user_input.lower() == 'quit':
        print("Goodbye! 👋")
        break


    # Add current user message to conversation history
    conversation.append({
        "role": "user",
        "content": user_input
    })


    # Send full conversation history to AI model
    # This gives context/memory to the chatbot
    response = ollama.chat(
        model="gemma3:2b",
        messages=conversation
    )


    # Extract only the text reply from Ollama response
    ai_reply = response['message']['content']


    # Store AI response in conversation history
    # So future prompts remember what AI said
    conversation.append({
        "role": "assistant",
        "content": ai_reply
    })


    # Display AI response on terminal
    print(f"\n🤖 AI: {ai_reply}")