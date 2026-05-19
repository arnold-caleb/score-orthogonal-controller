"""GSM8K-train dataset for REINFORCE: yields prompt-only tokenization + the
gold answer string for verifier scoring.

Differs from psi_dataset.GSM8KTrainDataset:
  - input_ids = BOS + question + separator (NO answer; the model generates it)
  - prompt_len = len(input_ids before padding)
  - gold_answer_str preserved for the verifier

Format per example:
  {
    'input_ids':       (L,) int64, padded with pad_id
    'prompt_len':      int
    'gold_answer_str': str  (the HF gsm8k 'answer' field, contains '#### N')
  }
"""
import os
import json
import torch
from torch.utils.data import Dataset


class GSM8KRLDataset(Dataset):
    # CRITICAL: separator default is the LITERAL two-character string '\n'
    # (backslash + n), which is what the YAML config encodes via
    # `separator: '\n'` (single-quoted YAML does NOT process escapes).
    # The frozen S-FLM model was trained against this literal separator;
    # passing the Python newline character would produce a different
    # token after the prompt and the model would never emit `def …`.
    def __init__(self, tokenizer, block_size, separator='\\n',
                 cache_path=None, hf_split='train', max_prompt_len=None):
        records = _load_gsm8k_pairs(cache_path, hf_split)
        BOS = tokenizer.bos_token_id
        EOS = tokenizer.eos_token_id
        PAD = tokenizer.pad_token_id
        if PAD is None:
            PAD = EOS
        sep_ids = tokenizer(separator, add_special_tokens=False).input_ids

        self.records = []
        self.n_dropped = 0
        cap = max_prompt_len or (block_size // 2)   # leave room for generation
        for r in records:
            q_ids = tokenizer(r['question'].strip(),
                              add_special_tokens=False).input_ids
            ids = [BOS] + q_ids + sep_ids
            prompt_len = len(ids)
            if prompt_len > cap:
                self.n_dropped += 1
                continue
            # Pad to block_size
            ids = ids + [PAD] * (block_size - prompt_len)
            self.records.append({
                'input_ids':       torch.tensor(ids, dtype=torch.long),
                'prompt_len':      prompt_len,
                'gold_answer_str': r['answer'],
            })
        self.block_size = block_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def _load_gsm8k_pairs(cache_path, hf_split):
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=hf_split)
    records = [{'question': ex['question'], 'answer': ex['answer']}
               for ex in ds]
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(records, f)
    return records
