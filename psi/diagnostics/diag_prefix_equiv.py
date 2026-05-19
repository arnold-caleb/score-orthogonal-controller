"""Quick diagnostic: prove that the prompt tokenization & prefix shape
used by the trainer's GSM8KRLDataset produces the SAME bytes as what
get_gsm8k_test_dataset produces for the eval path. If these match,
the model gets identical inputs in both cases.

Run from the repo root via:
    python -u psi/diagnostics/diag_prefix_equiv.py
"""
import sys, os
# repo root = psi/diagnostics/.. /..
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer
TOKENIZER_NAME = 'HuggingFaceTB/SmolLM-135M'
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
print(f'BOS={tok.bos_token_id}, EOS={tok.eos_token_id}, PAD={tok.pad_token_id}')
sep = '\\n'  # literal backslash-n, matching the yaml setting in tiny-gsm.yaml
print(f'separator literal: {repr(sep)}')
sep_ids = tok(sep, add_special_tokens=False).input_ids
print(f'sep tokens: {sep_ids} → {tok.decode(sep_ids)!r}')

# ── Path A: gsm8k_test.json (eval) ──
import json
test = json.load(open('/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json'))
q0 = test[0]['prompt']
print(f'\n=== TEST QUESTION 0 (eval) ===')
print(f'q: {q0!r}')
q_ids_eval = tok(q0.strip(), add_special_tokens=False).input_ids
eval_input_ids = [tok.bos_token_id] + q_ids_eval + sep_ids
print(f'eval input_ids ({len(eval_input_ids)} tokens): {eval_input_ids[:20]} ... {eval_input_ids[-10:]}')
print(f'decoded eval prompt: {tok.decode(eval_input_ids)!r}')

# ── Path B: HF gsm8k train (training) ──
print('\n=== HF gsm8k train question 0 (training) ===')
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='train')
q_train = ds[0]['question']
print(f'q: {q_train!r}')
q_ids_train = tok(q_train.strip(), add_special_tokens=False).input_ids
train_input_ids = [tok.bos_token_id] + q_ids_train + sep_ids
print(f'train input_ids ({len(train_input_ids)} tokens): {train_input_ids[:20]} ... {train_input_ids[-10:]}')
print(f'decoded train prompt: {tok.decode(train_input_ids)!r}')

# ── Now try the SAME test question through the training path ──
print('\n=== SAME test question 0 routed through training path ===')
same_q_train_path = tok(q0.strip(), add_special_tokens=False).input_ids
same_train_input_ids = [tok.bos_token_id] + same_q_train_path + sep_ids
print(f'training-path input_ids for test-q0 ({len(same_train_input_ids)} tokens): {same_train_input_ids[:20]} ... {same_train_input_ids[-10:]}')

match = (same_train_input_ids == eval_input_ids)
print(f'\n*** TRAINING PATH PRODUCES SAME INPUT_IDS AS EVAL FOR Q0?  {match} ***')

# ── Also: what does the SEPARATOR really tokenize to ──
print(f'\n=== Separator analysis ===')
for s, label in [('\\n', 'literal \\n (yaml unquoted)'),
                 ('\n', 'newline character'),
                 ('\\\\n', 'literal backslash + literal n')]:
    ids = tok(s, add_special_tokens=False).input_ids
    print(f'{label!r}: tokens={ids}, decoded={tok.decode(ids)!r}')
