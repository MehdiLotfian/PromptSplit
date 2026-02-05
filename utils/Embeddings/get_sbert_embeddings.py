import torch
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import SentenceTransformer
import json


def get_roberta_embeddings(texts):
    """
    Generates RoBERTa embeddings for a list of texts.
    """
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    mask = inputs['attention_mask']
    mask_expanded = mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled_embeddings = sum_embeddings / sum_mask
    return mean_pooled_embeddings

def get_sbert_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=64, device=None):
    """
    Generates Sentence-BERT embeddings for a list of texts.
    Returns a torch.Tensor of shape (len(texts), dim).
    """
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True)
    # move to CPU for consistent saving
    emb = emb.cpu()
    return emb

SENTENCES_FILE = f"LLM_experiments/gemma3_band_prompt_response_v1.json"
mode = 'json'  # 'txt', 'jsonl', or 'json'
texts = []
if mode == 'txt':
    with open(SENTENCES_FILE, "r", encoding="utf-8") as sf:
        texts = [l.strip() for l in sf if l.strip()]
elif mode == 'jsonl':
    with open(SENTENCES_FILE, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            if i >= 20000:
                break  # limit to 20k for speed
            i += 1
            texts.append(' '.join(json.loads(line)['answer']))
else:
    with open(SENTENCES_FILE, 'r') as f:
        data_list = json.load(f)
    for entry in data_list:
        for answer in entry['answers']:
            texts.append(answer)
print(f"Successfully loaded {len(texts)} entries from {SENTENCES_FILE}.")

print(f"Generating SBERT embeddings for {len(texts)} texts (high-quality model)...")
model_name = "all-mpnet-base-v2"  # high-quality SBERT
embeddings = get_sbert_embeddings(texts, model_name=model_name, batch_size=64, device="cuda" if torch.cuda.is_available() else "cpu")

# Save embeddings tensor only (no labels) for best quality experiments
out_path = f"LLM_experiments/gemma/band/embeddings.pt"
torch.save(embeddings, out_path)
print(f"Saved embeddings tensor {embeddings.shape} to {out_path}")
# --- 1. Define the prompts and generate embeddings ---