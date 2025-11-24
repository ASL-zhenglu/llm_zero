from transformers import AutoTokenizer
import json
# 没啥用
try:
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer len: {len(tokenizer)}")
    
    # Check if there are any tokens with ID >= 6400
    # We can check the vocab directly
    vocab = tokenizer.get_vocab()
    max_id = max(vocab.values())
    print(f"Max token ID in vocab: {max_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"Token 0: {tokenizer.decode([0])}")
    
except Exception as e:
    print(f"Error loading tokenizer: {e}")
