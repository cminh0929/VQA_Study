"""
Vocabulary Builder for VQA
Builds word-to-index mappings for questions and answers
"""

import json
from collections import Counter
from typing import List, Dict

class Vocabulary:
    """Vocabulary class for text tokenization"""
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'  # Start of sequence
    EOS_TOKEN = '<EOS>'  # End of sequence
    
    def __init__(self, min_word_freq: int = 1):
        """
        Args:
            min_word_freq: Minimum frequency for a word to be included
        """
        self.min_word_freq = min_word_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        
        # Initialize with special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN
        ]
        for token in special_tokens:
            self._add_word(token)
    
    def _add_word(self, word: str):
        """Add a word to vocabulary"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocab(self, texts: List[str]):
        """
        Build vocabulary from list of texts
        
        Args:
            texts: List of text strings
        """
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            self.word_freq.update(words)
        
        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= self.min_word_freq:
                self._add_word(word)
        
        print(f"Built vocabulary with {len(self.word2idx)} words")
        print(f"  - Min frequency: {self.min_word_freq}")
        print(f"  - Total unique words in corpus: {len(self.word_freq)}")
    
    def encode(self, text: str, max_length: int = None, add_sos_eos: bool = False) -> List[int]:
        """
        Convert text to list of indices
        
        Args:
            text: Input text
            max_length: Maximum sequence length (pad/truncate)
            add_sos_eos: Whether to add SOS/EOS tokens
        
        Returns:
            List of word indices
        """
        words = text.lower().split()
        indices = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
        
        # Add SOS/EOS tokens
        if add_sos_eos:
            indices = [self.word2idx[self.SOS_TOKEN]] + indices + [self.word2idx[self.EOS_TOKEN]]
        
        # Pad or truncate
        if max_length is not None:
            if len(indices) < max_length:
                # Pad
                indices += [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))
            else:
                # Truncate
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert list of indices back to text
        
        Args:
            indices: List of word indices
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text string
        """
        special_tokens = {
            self.word2idx[self.PAD_TOKEN],
            self.word2idx[self.UNK_TOKEN],
            self.word2idx[self.SOS_TOKEN],
            self.word2idx[self.EOS_TOKEN]
        }
        
        words = []
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        
        return ' '.join(words)
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'word_freq': dict(self.word_freq),
            'min_word_freq': self.min_word_freq
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"Saved vocabulary to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = cls(min_word_freq=vocab_data['min_word_freq'])
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab.word_freq = Counter(vocab_data['word_freq'])
        
        print(f"Loaded vocabulary from {filepath}")
        print(f"  - Vocabulary size: {len(vocab)}")
        return vocab


def build_vqa_vocabularies(
    train_json_path: str,
    question_vocab_path: str,
    answer_vocab_path: str,
    min_word_freq: int = 1
):
    """
    Build separate vocabularies for questions and answers
    
    Args:
        train_json_path: Path to train.json
        question_vocab_path: Output path for question vocabulary
        answer_vocab_path: Output path for answer vocabulary
        min_word_freq: Minimum word frequency
    """
    print("Building VQA vocabularies...")
    
    # Load training data
    print(f"\nLoading training data from {train_json_path}...")
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} Q&A pairs")
    
    # Extract questions and answers
    questions = [qa['question'] for qa in train_data]
    answers = [qa['answer'] for qa in train_data]
    
    # Build question vocabulary
    print("\nBuilding question vocabulary...")
    question_vocab = Vocabulary(min_word_freq=min_word_freq)
    question_vocab.build_vocab(questions)
    question_vocab.save(question_vocab_path)
    
    # Build answer vocabulary
    print("\nBuilding answer vocabulary...")
    answer_vocab = Vocabulary(min_word_freq=min_word_freq)
    answer_vocab.build_vocab(answers)
    answer_vocab.save(answer_vocab_path)
    
    print("Vocabulary building complete.")
    print(f"  Question vocab size : {len(question_vocab)}")
    print(f"  Answer vocab size   : {len(answer_vocab)}")
    
    return question_vocab, answer_vocab



