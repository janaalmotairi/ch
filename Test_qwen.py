from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


model_name="Qwen/Qwen2.5-1.5B"

tokenizer=AutoTokenizer.from_pretrained(model_name)

model=AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16
)

prompt="Give me a short introduction about yourself"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text=tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs=tokenizer([text], return_tensors="pt").to(model.device)

generated_ids=model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

generated_ids=[
    output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs.input_ids,generated_ids)
]

response=tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]

print(response)