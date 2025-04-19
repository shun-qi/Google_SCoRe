from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = "/data1/yx/plm/LLM-Research/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="auto")

# 组织 Prompt（Gemma 3 使用 chat template 格式）
messages = [
    {"role": "user", "content": [{"type": "text", "text": "朝鲜王朝的建国年是哪一年？"}]},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

outputs = model.generate(input_ids=input_ids, max_new_tokens=40)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)



import pandas as pd 

df = pd.read_csv("./SCoRe_Dasetset.csv")
df.tail(20)


from transformers import AutoTokenizer,AutoModelForCausalLM 
tokenizer = AutoTokenizer.from_pretrained("/data1/yx/plm/LLM-Research/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("/data1/yx/plm/LLM-Research/gemma-3-1b-it").to("cuda")



input_text = "조선시대의 대표적인 그림 양식은 무엇인가요?"


def change_inference_chat_format(input_text):
    return [
    {"role": "user", "content": f"{input_text}"},
    {"role": "assistant", "content": ""}
    ]
prompt = change_inference_chat_format(input_text)
# tokenizer 초기화 및 적용t\
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



input_text = "산성도를 나타내는 척도는 무엇인가요?"


def change_inference_chat_format(input_text):
    return [
    {"role": "user", "content": f"{input_text}"},
    {"role": "assistant", "content": ""}
    ]
prompt = change_inference_chat_format(input_text)
# tokenizer 초기화 및 적용t\
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


from transformers import AutoTokenizer,AutoModelForCausalLM 
tokenizer = AutoTokenizer.from_pretrained("/data1/yx/plm/LLM-Research/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("./trained_self_correcting_model").to("cuda")




input_text = "산성도를 나타내는 척도는 무엇인가요?"


def change_inference_chat_format(input_text):
    return [
    {"role": "user", "content": f"{input_text}"},
    {"role": "assistant", "content": ""}
    ]
prompt = change_inference_chat_format(input_text)
# tokenizer 초기화 및 적용t\
inputs = tokenizer.apply_chat_template(prompt, tokenize=True, 
                                       add_generation_prompt=True, 
                                       return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))