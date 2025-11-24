import torch
from transformers import AutoTokenizer
from dataset import SFTDataset
import numpy as np

def check_dataset():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    
    # Apply the fix that is in sft_train.py
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer len: {len(tokenizer)}")
    
    print("Loading dataset...")
    # Use a small max_seq_len for speed, or match the training one
    dataset = SFTDataset('./dataset/sft_mini_512.jsonl', tokenizer, max_seq_len=512)
    
    print(f"Dataset size: {len(dataset)}")
    
    max_id_found = -1
    min_id_found = 100000
    
    print("Iterating through dataset...")
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            input_ids = item['input_ids']
            labels = item['labels']
            
            # Check input_ids
            if input_ids.numel() > 0:
                curr_max = input_ids.max().item()
                curr_min = input_ids.min().item()
                max_id_found = max(max_id_found, curr_max)
                min_id_found = min(min_id_found, curr_min)
                
                if curr_max >= 6400:
                    print(f"Found out of bound token in sample {i}: {curr_max}")
                    print(f"Input IDs: {input_ids}")
                    break
                if curr_min < 0:
                    print(f"Found negative token in input_ids sample {i}: {curr_min}")
                    break
            
            # Check labels
            # Labels can be -100
            if labels.numel() > 0:
                # Filter out -100
                valid_labels = labels[labels != -100]
                if valid_labels.numel() > 0:
                    l_max = valid_labels.max().item()
                    if l_max >= 6400:
                        print(f"Found out of bound label in sample {i}: {l_max}")
                        break
                        
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            break
            
        if i % 100 == 0:
            print(f"Processed {i} samples...")

    print(f"Max ID found: {max_id_found}")
    print(f"Min ID found: {min_id_found}")
    print("Done.")

if __name__ == "__main__":
    check_dataset()
