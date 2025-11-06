# preprocessing/tokenizer.py

from transformers import AutoTokenizer

# --- BERT токенизаторды инициализациялау ---

# NEW:
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def tokenize_with_bert(text: str):
    """BERT арқылы токенизация"""
    tokens = tokenizer(text, return_offsets_mapping=True)
    token_texts = tokenizer.convert_ids_to_tokens(tokens["input_ids"])
    return token_texts

def process_with_bert_tokenizer(text: str):
    """Толық токенизация процесі және debug ақпарат"""
    tokens = tokenize_with_bert(text)
    print(f"[Tokenized Text]: {tokens}")
    return tokens
