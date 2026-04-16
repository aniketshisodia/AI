# ==========================================================
# LESSON 1: BASIC SINGLE PROMPT AI USING OLLAMA
# ==========================================================
#
# This program sends one question to the AI model and gets
# one response back.
#
# Workflow:
# 1. Send a prompt/question to Gemma 3 model.
# 2. Model processes the prompt.
# 3. Receive structured response from Ollama.
# 4. Extract only the generated answer text.
# 5. Print final answer on screen.
#
# This is a SINGLE TURN interaction,
# meaning the AI does NOT remember anything after responding.
# Every time you run the program, it starts fresh.
# ==========================================================

import ollama


# Send a chat request to the AI model
response = ollama.chat(

    # Specify which model to use
    model="gemma3",

    # Messages must be passed in list format
    # Each message contains:
    # - role → who is speaking
    # - content → actual message text
    messages=[
        {
            "role": "user",   # Indicates the message is from human/user
            "content": "What makes coffe taste good?"   # User's question
        }
    ]
)


# Extract only the AI-generated text from response dictionary
answer = response['message']['content']


# Print the final answer nicely
print(f"Answer: {answer}")