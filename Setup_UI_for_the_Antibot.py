# -------------------------------------------------------
# Phase 3: Streamlit UI for Antibot AI
# -------------------------------------------------------
# Purpose:
#   - Professional Infectious Disease Assistant interface
#   - Patient profile automatically injected into every query
#   - Uses FAISS vectorstore + HuggingFace Mistral LLM
# -------------------------------------------------------

import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# -------------------------
# Step 1: Load environment variables
# -------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# -------------------------
# Step 2: Initialize HuggingFace client
# -------------------------
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -------------------------
# Step 3: Initialize memory
# -------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------
# Step 4: Load FAISS Vectorstore
# -------------------------
if os.path.exists("faiss_index"):
    db = FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)
else:
    db = None

# -------------------------
# Step 5: Streamlit UI Layout
# -------------------------
st.set_page_config(page_title="💊 Antibot AI", layout="centered")
st.title("💊 Antibot AI - Infectious Disease Assistant")
st.markdown("Your intelligent assistant for infectious disease management and antibiotic guidance.")

# -------------------------
# Sidebar: Patient Profile Input
# -------------------------
with st.sidebar:
    st.header("🧾 Patient Profile")
    age = st.text_input("Age", value="45")
    gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    allergies = st.text_input("Allergies (comma-separated)", value="Penicillin")
    renal_function = st.selectbox("Renal Function", ["Normal", "Impaired", "Dialysis"], index=0)
    hepatic_function = st.selectbox("Hepatic Function", ["Normal", "Impaired", "Cirrhosis"], index=0)

# -------------------------
# Build Patient Profile String
# -------------------------
patient_profile_str = (
    f"Patient Profile:\n"
    f"- Age: {age}\n"
    f"- Gender: {gender}\n"
    f"- Allergies: {allergies}\n"
    f"- Renal function: {renal_function}\n"
    f"- Hepatic function: {hepatic_function}"
)

st.sidebar.markdown("### Current Patient Context")
st.sidebar.text(patient_profile_str)
st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Note: Antibot AI is advisory only. Final decisions must be made by a licensed clinician.")

# -------------------------
# Step 6: Define query function
# -------------------------
def query_antibot(question, patient_profile):
    """
    Send query to LLM with automatic patient profile injection.
    """
    augmented_query = f"{patient_profile}\n\nQuestion: {question}"

    response = client.chat_completion(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are Antibot AI, an infectious disease assistant providing evidence-based recommendations."},
            {"role": "user", "content": augmented_query}
        ],
        max_tokens=500
    )
    return response.choices[0].message["content"]

# -------------------------
# Step 7: Chat input and display
# -------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_question = st.chat_input("Type your medical query here...")

if user_question:
    with st.spinner("🤔 Thinking..."):
        bot_response = query_antibot(user_question, patient_profile_str)

    # Save conversation
    st.session_state["chat_history"].append(("You", user_question))
    st.session_state["chat_history"].append(("Antibot AI", bot_response))

# Display conversation history
st.markdown("### 💬 Conversation")
for sender, msg in st.session_state["chat_history"]:
    if sender == "You":
        st.markdown(f"**🧑 {sender}:** {msg}")
    else:
        st.markdown(f"**🤖 {sender}:** {msg}")

# Optional: Display patient snapshot in main UI
st.markdown("### 🧾 Patient Snapshot")
st.info(patient_profile_str)


