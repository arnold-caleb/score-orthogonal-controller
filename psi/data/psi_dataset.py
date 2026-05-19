"""Fast in-memory GSM8K-train dataset for psi training.

GSM8K train has ~7.5k (question, answer) pairs. We tokenize once at
startup (a few seconds), keep everything in CPU memory, and serve
tensors directly. No disk cache, no HF datasets machinery.

Output per example matches what `tiny_gsm` with train_on_prompt=False,
wrap=False produces, so the same training loop works:
  - input_ids:      (L,) int64, padded with pad_id
  - attention_mask: (L,) int64, 1 ONLY on answer tokens (and the EOS
                    right after the answer), 0 on prompt + padding
"""
import os
import json
import torch
from torch.utils.data import Dataset


class GSM8KTrainDataset(Dataset):
    # Default separator is the LITERAL two-character '\n' (backslash+n)
    # to match the YAML config and what the frozen model was trained on.
    def __init__(self, tokenizer, block_size, separator='\\n',
                 cache_path=None, hf_split='train'):
        # 1. Get raw pairs
        records = _load_gsm8k_train_pairs(cache_path, hf_split)
        # 2. Tokenize and pad
        BOS = tokenizer.bos_token_id
        EOS = tokenizer.eos_token_id
        PAD = tokenizer.pad_token_id
        if PAD is None:
            PAD = EOS  # SmolLM has no PAD by default; reuse EOS like the SFM repo
        sep_ids = tokenizer(separator, add_special_tokens=False).input_ids

        self.records = []
        n_dropped = 0
        for r in records:
            q_ids = tokenizer(r['question'].strip(),
                              add_special_tokens=False).input_ids
            a_ids = tokenizer(r['answer'].strip(),
                              add_special_tokens=False).input_ids
            ids = [BOS] + q_ids + sep_ids + a_ids + [EOS]
            prompt_len = 1 + len(q_ids) + len(sep_ids)
            if len(ids) > block_size:
                # truncate answer rather than drop; keep last EOS
                ids = ids[:block_size - 1] + [EOS]
                if prompt_len >= block_size:
                    n_dropped += 1
                    continue
            # pad
            n = len(ids)
            ids = ids + [PAD] * (block_size - n)
            # answer mask: 1 on [prompt_len, n), 0 elsewhere
            mask = [0] * prompt_len + [1] * (n - prompt_len) + [0] * (block_size - n)
            self.records.append({
                'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(mask, dtype=torch.long),
            })
        self.block_size = block_size
        self.n_dropped = n_dropped

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def _load_gsm8k_train_pairs(cache_path, hf_split):
    """Returns list of {'question': str, 'answer': str}."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    # Fall back to HF datasets
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=hf_split)
    records = [{'question': ex['question'], 'answer': ex['answer']}
               for ex in ds]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(records, f)
    return records
