# ---------------------------------------------------------
# Phase 2: Connect Memory with LLM
# Project: Antibot AI (Infectious Disease Physician Assistant)
# Purpose:
#   - Connects patient-specific memory (from Phase 1) with a Large Language Model (LLM).
#   - Uses Hugging Face Inference API to query the "mistralai/Mistral-7B-Instruct-v0.2" model.
#   - Handles conversation in chat mode (conversational task).
#   - Stores conversation context for continuity across multiple queries.
# ---------------------------------------------------------

from huggingface_hub import InferenceClient
from Setup_memory_for_llm import memory  # Import patient context + conversation memory

# ---------------------------------------------------------
# Step 1: Initialize HuggingFace Inference Client
# ---------------------------------------------------------
# Mistral-7B-Instruct-v0.2 is designed for chat/conversational tasks,
# not raw text generation. Therefore, we must use `chat_completion`.
# ---------------------------------------------------------
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # Chat-capable model
)

print("🔗 Connecting Infectious Disease Assistant...")


# ---------------------------------------------------------
# Step 2: Function to Query Antibot
# ---------------------------------------------------------
def query_antibot(user_question: str) -> str:
    """
    Purpose:
        - Takes a user question as input.
        - Retrieves past conversation history from memory.
        - Sends conversation (context + question) to the LLM in chat format.
        - Gets a medically relevant response.
        - Updates memory with the new Q&A for continuity.
    """

    # Load conversation history from memory (Phase 1 output)
    history = memory.load_memory_variables({}).get("chat_history", [])

    # Convert memory history into Hugging Face chat format
    messages = []
    for msg in history:
        if hasattr(msg, "content"):
            # Identify message role (user or assistant)
            role = "user" if msg.__class__.__name__ == "HumanMessage" else "assistant"
            messages.append({"role": role, "content": msg.content})

    # Add the latest user query
    messages.append({"role": "user", "content": user_question})

    # ---------------------------------------------------------
    # Step 3: Send request to Hugging Face ChatCompletion API
    # ---------------------------------------------------------
    response = client.chat_completion(
        messages=messages,   # Conversation so far
        max_tokens=500,      # Limit to avoid overly long answers
        temperature=0.7,     # Balanced creativity vs factuality
    )

    # Extract the assistant's answer
    ai_answer = response.choices[0].message["content"]

    # ---------------------------------------------------------
    # Step 4: Update Memory with New Interaction
    # ---------------------------------------------------------
    memory.chat_memory.add_user_message(user_question)
    memory.chat_memory.add_ai_message(ai_answer)

    return ai_answer


# ---------------------------------------------------------
# Step 5: Test Run (Standalone Execution)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example realistic query for infectious disease context
    user_question = "What is the best antibiotic for a diabetic patient with pneumonia?"
    answer = query_antibot(user_question)

    print("\n💊 Antibot Response:")
    print(answer)









