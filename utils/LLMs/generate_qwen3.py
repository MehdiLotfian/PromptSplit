from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch

model_name = "Qwen/Qwen3-1.7B"

file_path = "/research/d7/gds/lotfian25/CVPR2026/LLM_experiments/NQ_Open/NQ-open.train.jsonl"
out_file_path = '/research/d7/gds/lotfian25/CVPR2026/LLM_experiments/NQ_Open/qwen/qwen3_nq_open_prompts_responses.json'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"                # <-- minimal change: left padding for decoder-only
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare prompts from a JSON file (assumes a JSON list of strings)
prompts = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        prompts.append(json.loads(line)['question'])
prompts = prompts[:20000]  # limit to 10

results = []
batch_size = 64
for i in tqdm(range(0, len(prompts), batch_size)):
    print(f"\n[LOG] Processing Prompt {i + 1}/{len(prompts)}")
    batch = prompts[i : i + batch_size]
    texts = []
    for prompt in batch:
        messages = [{"role": "user", "content": prompt + 'answer very very short'}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,   # reduce from huge value for speed; increase if needed
            do_sample=False,      # deterministic; set True + temp/top_p for sampling
        )

    input_lens = model_inputs["attention_mask"].sum(dim=1).tolist()
    for j, prompt in enumerate(batch):
        output_ids = generated_ids[j][input_lens[j]:].tolist()
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        results.append({"prompt": prompt, "content": content})

    if (i // batch_size) % 10 == 0:
        print(f"processed {min(i+batch_size, len(prompts))}/{len(prompts)} prompts")


with open(out_file_path, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=2)