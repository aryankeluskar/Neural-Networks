from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers

m_name = "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
model = AutoModelForCausalLM.from_pretrained(
    m_name,
    device_map="auto",
    trust_remote_code=False,
    revision="main")
tokenizer = AutoTokenizer.from_pretrained(m_name, use_fast=True)

model.eval()
comment = "This video looks great!"
prompt = f'''[INST] {comment} [/INST]'''

input = tokenizer(prompt, return_tensors="pt")
output = model.generate(input["input_ids"].to("cuda"), max_new_tokens=140)

print(tokenizer.batch_decode(output)[0])