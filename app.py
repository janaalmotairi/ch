import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from RAG import get_context 

st.set_page_config(page_title="HR Chatbot", page_icon="🤖")
st.title("HR Assistant Chatbot 🤖")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Select Model",
        ("Local Qwen 1.5B", "Cloud Qwen ......")
    )
    st.markdown("---")

    # Cache the model loading to prevent reloading on every interaction
    @st.cache_resource
    def load_local_model():
        # Point to the local folder 
        model_path = "./my_qwen_model" 
        
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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:   
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Handler
if prompt := st.chat_input("Ask your HR related questions here..."):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # --- Step 1: RAG (Get Context) ---
        context_text = ""
        try:
            # Using get_context as imported
            results = get_context(prompt)
            
            # Assuming get_context returns a list of documents (standard LangChain)
            if isinstance(results, list):
                context_text = "\n".join([doc.page_content for doc in results])
            else:
                # If it returns a string directly
                context_text = str(results)

        except Exception as e:
            st.error(f"Error connecting to Knowledge Base: {e}")

        # Prepare the prompt with context
        final_prompt = f"Answer based on this context:\n{context_text}\n\nQuestion:{prompt}"

        # --- Step 2: Model Generation ---
        full_response = "" 
        
        if model_name == "Local Qwen 1.5B":
            model, tokenizer = load_local_model()
            
            if model and tokenizer:
                messages = [
                    {"role": "system", "content": "You are a helpful HR assistant."},
                    {"role": "user", "content": final_prompt}
                ]
                
                # Apply chat template
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                model_input = tokenizer(
                    [text],
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(model.device)
                
                generated_ids = model.generate(
                    model_input.input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=0.7,
                    temperature=0.95
                )
                
                # Decode output (remove input tokens)
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(model_input.input_ids, generated_ids)
                ]
                full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        else:
            full_response = "............."

        # Display the response to the user
        message_placeholder.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})