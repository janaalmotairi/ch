import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from RAG import get_context 
import os

st.set_page_config(page_title="HR Chatbot", page_icon="🤖")
st.title("HR Assistant Chatbot 🤖")

# --- Function to save chat history to a text file ---
def save_to_file(role, text):
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"{role.upper()}:\n{text}\n\n")
        f.write("----------------------\n\n")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Model",
        ("Local Qwen 1.5B", "Cloud Qwen 72B")
    )
    st.markdown("---")

    #SECURE_API_TOKEN = "**********"

    st.markdown("Chat History")

    # Display history and Clear Button
    if os.path.exists("chat_history.txt"):
        with open("chat_history.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
            # Display history in an expander
            if lines:
                with st.expander("View History"):
                    for line in lines:
                        if line.strip():
                            st.caption(line.strip())
                
                # Button to clear history
                if st.button("Clear Chat History", key="clear_history_btn"):
                    open("chat_history.txt", "w").close() # Clears the file
                    st.rerun()
            else:
                st.info("History is empty.")
    else:
        st.info("No chat history found.")
                    
    st.markdown("---")
    st.markdown("Developed by Jana 👩🏻‍💻")

    @st.cache_resource
    def load_local_model():
        model_path = "./local_qwen_model" 
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            return model, tokenizer
        except OSError:
            return None, None

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Previous Messages ---
for message in st.session_state.messages:   
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle New User Input ---
if prompt := st.chat_input("Ask your HR related questions here..."):
    
    # 1. Display and save user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_to_file("user", prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # A. Retrieve Context from RAG
        context_text = ""
        try:
            results = get_context(prompt)
            if isinstance(results, list):
                context_text = "\n".join([doc.page_content for doc in results])
            else:
                context_text = str(results)
        except Exception as e:
            st.error(f"Error connecting to Knowledge Base: {e}")

        final_prompt = f"Answer based on this context:\n{context_text}\n\nQuestion:{prompt}"

        full_response = "" 
        
        # B. Choose between Local and Cloud Model
        if model_name == "Local Qwen 1.5B":
            # --- Run Local Model ---
            model, tokenizer = load_local_model()
            
            if model and tokenizer:
                messages = [
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": final_prompt}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                model_input = tokenizer([text], return_tensors="pt").to(model.device)
                
                generated_ids = model.generate(
                    model_input.input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7
                )
                
                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)]
                full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                full_response = "Error: Local model not found."

        else:
            # --- Run Cloud Model (Cloud Qwen 72B) ---
            try:
                client = InferenceClient(token=SECURE_API_TOKEN)
                response = client.chat_completion(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful HR assistant."},
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=512,
                    stream=False
                )
                full_response = response.choices[0].message.content
            except Exception as e:
                st.error(f"Error loading cloud model: {e}")
                full_response = "Cloud Error."

        # 3. Display and save assistant response
        message_placeholder.markdown(full_response)
        save_to_file("assistant", full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})