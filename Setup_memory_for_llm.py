# ---------------------------------------------------------
# Phase 1: Setup_memory_for_llm.py
# ---------------------------------------------------------
# Purpose:
# This script sets up the memory system for the Antibot AI.
# The memory not only stores conversation history but also
# includes structured patient information (age, gender, 
# allergies, renal/hepatic function). This design enables 
# more context-aware and patient-specific responses from the LLM.
# ---------------------------------------------------------

from langchain.memory import ConversationBufferMemory

# ---------------------------------------------------------
# Step 1: Define structured patient context
# ---------------------------------------------------------
# We create a dictionary to hold patient-related details.
# This ensures that every conversation retains crucial 
# medical context, simulating how an infectious disease 
# physician keeps patient notes.
# ---------------------------------------------------------
patient_context = {
    "age": None,              # Patient's age (int)
    "gender": None,           # Patient's gender (M/F/Other)
    "allergies": None,        # Known drug allergies
    "renal_function": None,   # e.g., "Normal", "Impaired", "Dialysis"
    "hepatic_function": None  # e.g., "Normal", "Impaired", "Cirrhosis"
}

# ---------------------------------------------------------
# Step 2: Setup conversational memory
# ---------------------------------------------------------
# ConversationBufferMemory is used to store dialogue history.
# We extend it by embedding the patient context, ensuring 
# that the chatbot retains both conversational and clinical details.
# ---------------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",      # Stores Q&A dialogue
    input_key="question",           # User input
    output_key="answer",            # Bot output
    return_messages=True            # Returns full chat history
)

# ---------------------------------------------------------
# Step 3: Function to update patient context
# ---------------------------------------------------------
# This allows dynamic updates of patient details during 
# the session. For example, if the user later specifies 
# “Patient has penicillin allergy”, the context updates.
# ---------------------------------------------------------
def update_patient_context(age=None, gender=None, allergies=None,
                           renal_function=None, hepatic_function=None):
    if age:
        patient_context["age"] = age
    if gender:
        patient_context["gender"] = gender
    if allergies:
        patient_context["allergies"] = allergies
    if renal_function:
        patient_context["renal_function"] = renal_function
    if hepatic_function:
        patient_context["hepatic_function"] = hepatic_function

# ---------------------------------------------------------
# Step 4: Helper to fetch current patient context
# ---------------------------------------------------------
# This function provides the stored patient profile in a 
# human-readable way, which can be injected into the 
# system prompt when querying the LLM.
# ---------------------------------------------------------
def get_patient_context():
    return {
        "age": patient_context["age"],
        "gender": patient_context["gender"],
        "allergies": patient_context["allergies"],
        "renal_function": patient_context["renal_function"],
        "hepatic_function": patient_context["hepatic_function"]
    }

# ---------------------------------------------------------
# Step 5: Example usage (for testing only)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Example: Updating patient info
    update_patient_context(age=45, gender="Male", allergies="Penicillin",
                        renal_function="Impaired", hepatic_function="Normal")
    
    # Fetch and print patient profile
    print("Current Patient Context:")
    print(get_patient_context())
    
    # Example: Storing a conversation
    memory.save_context({"question": "What antibiotic for pneumonia?"}, 
                        {"answer": "Refer to Antibiotics Simplified, Chapter 5."})
    
    print("\nConversation Memory:")
    print(memory.load_memory_variables({}))


