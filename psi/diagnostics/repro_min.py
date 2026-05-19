"""Minimal reproducer: load model exactly as _sample_gsm8k does, then
call model.generate_samples on a single GSM8K test prompt. See whether
it produces Python code (baseline-quality) or gibberish."""
import sys, os
# repo root = psi/diagnostics/.. /..
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch as _torch_compat
if not hasattr(_torch_compat.library, 'wrap_triton'):
    _torch_compat.library.wrap_triton = lambda fn: fn

import json, torch
import hydra
import lightning as L
import dataloader as dataloader_mod
import algo

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    tokenizer = dataloader_mod.get_tokenizer(config)

    # Same as _load_from_checkpoint + _sample_gsm8k init
    model = algo.SFM.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer, config=config,
        strict=config.eval.strict_loading)
    model.to('cuda')
    if config.eval.disable_ema: model.ema = None
    model._eval_mode()

    # Test prompt
    test = json.load(open('/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json'))
    rec = test[0]
    q_ids = tokenizer(rec['prompt'].strip(),
                      add_special_tokens=False).input_ids
    sep_ids = tokenizer(config.data.separator,
                        add_special_tokens=False).input_ids
    prompt_ids = [tokenizer.bos_token_id] + q_ids + sep_ids
    prompt_len = len(prompt_ids)

    block = int(config.model.length)
    # PATH A: eval-style — prefix_tokens shape (1, prompt_len)
    ids_eval = torch.tensor(prompt_ids, dtype=torch.long,
                            device='cuda').unsqueeze(0)
    plen = torch.tensor([prompt_len], device='cuda')

    # PATH B: trainer-style — pad to block_size with PAD (EOS for SmolLM)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded = prompt_ids + [pad_id] * (block - prompt_len)
    ids_train = torch.tensor(padded, dtype=torch.long,
                             device='cuda').unsqueeze(0)

    # Replicate the trainer's batch: 2 prompts × 4 replicas = 8 rows
    rec2 = test[1]
    q2 = tokenizer(rec2['prompt'].strip(),
                   add_special_tokens=False).input_ids
    p2 = [tokenizer.bos_token_id] + q2 + sep_ids
    pl2 = len(p2)
    pad_train2 = p2 + [pad_id] * (block - pl2)
    # Stack: [ids_train, pad_train2] then repeat_interleave 4×
    ids_stack = torch.stack([
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(pad_train2, dtype=torch.long)], 0).to('cuda')
    plens_stack = torch.tensor([prompt_len, pl2], device='cuda')
    K = 4
    ids_batched = ids_stack.repeat_interleave(K, dim=0)
    plens_batched = plens_stack.repeat_interleave(K)
    print(f'\n=== TRAINER-STYLE BATCH (B=2, K=4 = 8 rows) ===')
    print(f'  prefix_tokens.shape = {ids_batched.shape}')
    print(f'  prompt_lens = {plens_batched.cpu().tolist()}')
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokens, meta = model.generate_samples(
        num_samples=ids_batched.shape[0], num_steps=config.sampler.steps,
        prefix_tokens=ids_batched, prefix_lengths=plens_batched)
    for i in range(ids_batched.shape[0]):
        pl_i = int(plens_batched[i].item())
        out_ids = tokens[i, pl_i:].cpu().tolist()
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(f'  [row {i}, plen={pl_i}] RESPONSE: {text[:200]!r}')

    # And the original single-sample test as a control
    for label, ids, p in [('EVAL-STYLE (no pad, B=1)', ids_eval, plen),
                          ('TRAINER-STYLE (pad-to-block, B=1)', ids_train, plen)]:
        print(f'\n=== {label} ===')
        print(f'  prefix_tokens.shape = {ids.shape}, prompt_len = {int(p[0])}')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        tokens, meta = model.generate_samples(
            num_samples=1, num_steps=config.sampler.steps,
            prefix_tokens=ids, prefix_lengths=p)
        out_ids = tokens[0, int(p[0]):].cpu().tolist()
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(f'  RESPONSE: {text[:200]!r}')

if __name__ == '__main__':
    main()
