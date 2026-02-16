import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        
        # 1️⃣ Reset dictionaries
        self.word_to_id = {}
        self.id_to_word = {}
        
        # 2️⃣ Add special tokens with fixed IDs
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        # 3️⃣ Collect unique words from training texts
        unique_words = set()
        
        for text in texts:
            words = text.split()
            for word in words:
                if word not in self.word_to_id:  # Avoid re-adding special tokens
                    unique_words.add(word)
        
        # 4️⃣ Sort alphabetically for deterministic ordering
        sorted_words = sorted(unique_words)
        
        # 5️⃣ Assign IDs starting after special tokens
        current_id = len(special_tokens)
        
        for word in sorted_words:
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] = word
            current_id += 1
        
        # 6️⃣ Store vocabulary size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        Add BOS at start and EOS at end.
        """
        
        tokens = [self.word_to_id[self.bos_token]]
        
        words = text.split()
        
        for word in words:
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                tokens.append(self.word_to_id[self.unk_token])
        
        tokens.append(self.word_to_id[self.eos_token])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Remove BOS and EOS tokens.
        """
        
        words = []
        
        for idx in ids:
            word = self.id_to_word.get(idx, self.unk_token)
            
            # Skip BOS and EOS
            if word in [self.bos_token, self.eos_token]:
                continue
            
            words.append(word)
        
        return " ".join(words)
