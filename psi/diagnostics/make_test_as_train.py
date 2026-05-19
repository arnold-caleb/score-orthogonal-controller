"""Convert gsm8k_test.json into the HF gsm8k 'train' record format
({'question', 'answer'}) so we can route it through GSM8KRLDataset
as a sanity test for the trainer pipeline."""
import json, os
src = json.load(open('/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test.json'))
out = [{'question': r['prompt'], 'answer': r['response_ground_truth']}
       for r in src]
dst = '/n/fs/aa-rldiff/winter/s-flm/data/gsm8k_test_as_train.json'
os.makedirs(os.path.dirname(dst), exist_ok=True)
json.dump(out, open(dst, 'w'))
print(f'wrote {len(out)} records → {dst}')
