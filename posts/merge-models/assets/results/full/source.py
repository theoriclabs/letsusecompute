"""Train two Qwen3 adapters, export parents, and compare mergekit merges.

Self-contained synthetic data; response-only supervised fine-tuning. No API calls
for labels. All models see identical validation/test prompts at evaluation.
"""
from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import compute

app = compute.App('merge-models')
image = compute.Image.cuda_pytorch().pip_install(
    'transformers==4.51.3', 'peft==0.15.2', 'mergekit==0.1.4',
    'accelerate==1.6.0', 'huggingface_hub==0.30.2', 'datasets==3.5.0')
hf_secret = compute.Secret.from_name('hf')
BASE = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def make_data(sample=False):
    """Split templates and entity identifiers before generation; no row overlap."""
    intents = {
        'RUBY': ['charged twice', 'refund missing', 'invoice incorrect', 'payment declined'],
        'JADE': ['password forgotten', 'account locked', 'login code missing', 'sign-in blocked'],
        'AMBER': ['parcel delayed', 'package missing', 'delivery address wrong', 'shipment damaged'],
    }
    templates = {
        'train': ['Ticket {i}: {problem}.', 'Customer {i} reports: {problem}.',
                  'Please handle case {i}; {problem}.', 'Case {i} concerns: {problem}.'],
        'validation': ['For request {i}, the customer says: {problem}.'],
        'test': ['New support request #{i} — {problem}. Please assign it.',
                 'The issue in record {i} is "{problem}". Route this request.'],
    }
    extraction = {
        'train': ['Order {i}: {qty} units of SKU {sku}.',
                  'SKU {sku}; units requested: {qty}; reference {i}.',
                  'Please pick {qty} pieces, item {sku}, order {i}.',
                  'Reference {i} requests product {sku} with quantity {qty}.'],
        'validation': ['Request {i}: send item {sku}, {qty} units in total.'],
        'test': ['For order {i}, the requested quantity is {qty} and the SKU is {sku}.',
                 'Dispatch note {i}: product code {sku}; ship {qty} pieces.'],
    }
    data = {}
    for split, offset, n in [('train', 0, 60 if sample else 480),
                             ('validation', 10000, 6 if sample else 24),
                             ('test', 20000, 6 if sample else 48)]:
        rows = []
        for j in range(n):
            code = list(intents)[j % 3]
            problem = intents[code][(j // 3) % 4]
            ticket = templates[split][(j // 12) % len(templates[split])].format(i=offset+j, problem=problem)
            rows.append({'skill':'triage', 'prompt':'Assign the support queue. Reply with only the queue code.\n'+ticket, 'answer':code})
            sku = f'{["AX", "BQ", "CZ", "DP"][j % 4]}-{offset+j+100:05d}'
            qty = 1 + (j * 7) % 40
            order = extraction[split][j % len(extraction[split])].format(i=offset+j, sku=sku, qty=qty)
            rows.append({'skill':'extract', 'prompt':'Extract the order. Reply with only JSON containing sku (string) and qty (integer).\n'+order,
                         'answer':json.dumps({'sku':sku,'qty':qty}, separators=(',', ':'))})
        random.Random(42).shuffle(rows)
        data[split] = rows
    return data


def correct(row, text):
    text = text.strip()
    if row['skill'] == 'triage':
        return text == row['answer']
    try:
        got, expected = json.loads(text), json.loads(row['answer'])
        return isinstance(got, dict) and set(got) == {'sku','qty'} and type(got['qty']) is int and got == expected
    except (ValueError, TypeError):
        return False


def _train(sample=False, epochs=3, push=False):
    import gc
    import hashlib
    import importlib.metadata
    import subprocess
    import tempfile
    import torch
    import yaml
    from huggingface_hub import snapshot_download, HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    if sample and push:
        raise ValueError('Sample runs must not overwrite the published full model')
    if not 1 <= epochs <= 10:
        raise ValueError('epochs must be between 1 and 10')
    started = time.time()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    random.seed(42)
    assert torch.cuda.is_available(), 'This entrypoint requires a CUDA GPU'
    token = os.environ.get('HF_TOKEN') or os.environ.get('hf') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    work = Path(tempfile.mkdtemp(prefix='merge-models-'))
    out = Path(os.environ.get('COMPUTE_ARTIFACT_DIR', '/tmp/compute-artifacts')) / 'merge-models'
    out.mkdir(parents=True, exist_ok=True)
    (out / '.compute-artifact.json').write_text(json.dumps({'name':'qwen-merge-experiment', 'version':1,
        'kind':'output', 'compatibility_key':'qwen3-0.6b-merge', 'metadata':{'base':BASE,'revision':REVISION}}))
    data = make_data(sample)
    for split, rows in data.items():
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert not ({x['prompt'] for x in data['train']} & {x['prompt'] for x in data['test']+data['validation']})
    base_path = snapshot_download(BASE, revision=REVISION, token=token,
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE*','*.jinja'])
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    def prompt_text(row):
        return tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def load(path):
        return AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.bfloat16,
            attn_implementation='sdpa').to('cuda')

    def evaluate(model, rows):
        model.eval()
        results = []
        with torch.inference_mode():
            for start in range(0, len(rows), 12):
                batch = rows[start:start+12]
                inputs = tokenizer([prompt_text(r) for r in batch], padding=True, return_tensors='pt').to('cuda')
                generated = model.generate(**inputs, max_new_tokens=48, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, temperature=None, top_p=None, top_k=None)
                outputs = tokenizer.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                results.extend(dict(r, output=s, correct=correct(r,s)) for r,s in zip(batch,outputs))
        return {'scores':{skill:{'correct':sum(r['correct'] for r in results if r['skill']==skill),
                                'total':sum(r['skill']==skill for r in results)} for skill in ('triage','extract')},
                'examples':results}

    results = {'base':BASE, 'revision':REVISION,'sample':sample,'epochs':1 if sample else epochs,
               'gpu':torch.cuda.get_device_name(), 'versions':{p:importlib.metadata.version(p) for p in
                 ['torch','transformers','peft','mergekit','accelerate','huggingface_hub']},
               'data_sha256':{split:hashlib.sha256((out/f'{split}.jsonl').read_bytes()).hexdigest() for split in data},
               'training':{}, 'models':{}, 'merge_configs':{}}
    def save_results():
        (out/'results.json').write_text(json.dumps(results, indent=2))
    model = load(base_path)
    results['base_parameters'] = sum(p.numel() for p in model.parameters())
    for split in ('validation','test'):
        results['models'].setdefault('base',{})[split] = evaluate(model,data[split])
    print('BASE',json.dumps({s:results['models']['base'][s]['scores'] for s in ('validation','test')}),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

    for skill in ('triage','extract'):
        model = get_peft_model(load(base_path), LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            task_type='CAUSAL_LM'))
        model.config.use_cache = False
        trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
        before = {n:p.detach().cpu().clone() for n,p in trainable.items()}
        records = []
        for row in data['train']:
            if row['skill'] != skill: continue
            prefix = tokenizer.encode(prompt_text(row), add_special_tokens=False)
            answer = tokenizer.encode(row['answer']+tokenizer.eos_token, add_special_tokens=False)
            ids = prefix+answer
            assert len(ids)<=192, 'No silent truncation'
            records.append((ids, [-100]*len(prefix)+answer))
        optimizer = torch.optim.AdamW(trainable.values(),lr=2e-4,weight_decay=0.01)
        losses=[]; max_grad=0.0
        for epoch in range(1 if sample else epochs):
            random.Random(42+epoch).shuffle(records)
            model.train()
            epoch_loss=0.; batches=0
            for start in range(0,len(records),8):
                batch=records[start:start+8]; length=max(len(x[0]) for x in batch)
                # Right pad training; left pad generation. Labels ignore all prompt/padding tokens.
                ids=torch.tensor([x+[tokenizer.pad_token_id]*(length-len(x)) for x,y in batch],device='cuda')
                labels=torch.tensor([y+[-100]*(length-len(y)) for x,y in batch],device='cuda')
                mask=torch.tensor([[1]*len(x)+[0]*(length-len(x)) for x,y in batch],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
                assert torch.isfinite(loss)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(trainable.values(),1.0)
                max_grad=max(max_grad,float(norm))
                optimizer.step()
                value=float(loss.detach());losses.append(value);epoch_loss+=value;batches+=1
            print('TRAIN',skill,'epoch',epoch+1,'loss',epoch_loss/batches,flush=True)
        changed=sum(not torch.equal(before[n],p.detach().cpu()) for n,p in trainable.items())
        assert max_grad>0 and changed>0
        adapter=out/f'adapter-{skill}'; model.save_pretrained(adapter);tokenizer.save_pretrained(adapter)
        model.config.use_cache=True
        expected=evaluate(model,data['validation'][:6])['examples']
        del model, optimizer, trainable, before;gc.collect();torch.cuda.empty_cache()
        model=PeftModel.from_pretrained(load(base_path),adapter)
        actual=evaluate(model,data['validation'][:6])['examples']
        reload_ok=[r['output'] for r in actual]==[r['output'] for r in expected]
        assert reload_ok
        # Merge LoRA in float32 to avoid rounding away small weight updates.
        model=model.float().merge_and_unload()
        parent=work/f'parent-{skill}'; model.save_pretrained(parent);tokenizer.save_pretrained(parent)
        model=model.to(dtype=torch.bfloat16)
        results['training'][skill]={'examples':len(records),'losses':losses,'max_gradient_norm':max_grad,
            'changed_tensors':changed,'reload_ok':reload_ok}
        # Exact trainable count comes from saved adapter tensors, not a model-size estimate.
        from safetensors.torch import load_file
        results['training'][skill]['trainable_parameters']=sum(t.numel() for t in load_file(str(adapter/'adapter_model.safetensors')).values())
        results['models'][skill]={s:evaluate(model,data[s]) for s in ('validation','test')}
        print('PARENT',skill,json.dumps({s:results['models'][skill][s]['scores'] for s in ('validation','test')}),flush=True)
        save_results()
        del model;gc.collect();torch.cuda.empty_cache()

    configs = {
        'linear-balanced':{'merge_method':'linear','models':[
            {'model':str(work/'parent-triage'),'parameters':{'weight':0.5}},
            {'model':str(work/'parent-extract'),'parameters':{'weight':0.5}}], 'parameters':{'normalize':True}},
        'linear-skewed':{'merge_method':'linear','models':[
            {'model':str(work/'parent-triage'),'parameters':{'weight':0.95}},
            {'model':str(work/'parent-extract'),'parameters':{'weight':0.05}}], 'parameters':{'normalize':True}},
        'ties':{'merge_method':'ties','base_model':base_path,'models':[
            {'model':str(work/'parent-triage'),'parameters':{'weight':0.5,'density':0.2}},
            {'model':str(work/'parent-extract'),'parameters':{'weight':0.5,'density':0.2}}],
            'parameters':{'normalize':True}},
    }
    for name,config in configs.items():
        config.update(dtype='float32')
        config_path=out/f'{name}.yaml';config_path.write_text(yaml.safe_dump(config))
        results['merge_configs'][name]=config
        merged=work/name
        subprocess.run(['mergekit-yaml',str(config_path),str(merged),'--cuda'],check=True)
        model=load(merged)
        results['models'][name]={s:evaluate(model,data[s]) for s in ('validation','test')}
        print('MERGE',name,json.dumps({s:results['models'][name][s]['scores'] for s in ('validation','test')}),flush=True)
        save_results(); del model;gc.collect();torch.cuda.empty_cache()
    # Select by validation only: maximize worse skill, then summed accuracy.
    def rank(name):
        scores=results['models'][name]['validation']['scores']
        acc=[v['correct']/v['total'] for v in scores.values()]
        return min(acc),sum(acc)
    best=max(configs,key=rank)
    results['selected_by_validation']=best
    model=load(work/best)
    best_path=out/'best-merge';model.save_pretrained(best_path);tokenizer.save_pretrained(best_path)
    del model;gc.collect();torch.cuda.empty_cache()
    model=load(best_path)
    verified=evaluate(model,data['test'])
    results['merge_reload_ok']=[r['output'] for r in verified['examples']]==[r['output'] for r in results['models'][best]['test']['examples']]
    assert results['merge_reload_ok']
    results['elapsed_seconds']=time.time()-started
    results['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
    results['hub_repo']=None
    card='''---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-0.6B\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- mergekit\n- synthetic-data\n---\n# Qwen3 support and order merge\n\nA small synthetic experiment, not a production support model. Two independently trained LoRA adapters learn custom queue codes (RUBY=billing, JADE=login, AMBER=shipping) and SKU/quantity JSON extraction. Training templates and entity IDs are disjoint from evaluation. Intent phrases recur: this tests narrow formatting/label adaptation, not broad language generalization.\n\nDisable thinking via `enable_thinking=False` in the chat template. Greedy decode, 48 new tokens. Queue scoring is exact; extraction requires only sku/string and qty/integer and exact values.\n\nSelected on validation by worst-skill accuracy, then total; no test-set selection. Scores, all prompts and outputs, dependency versions and training evidence are in results.json. Reproduce with the bundled train.py. The experiment guide is being prepared in https://github.com/theoriclabs/letsusecompute/tree/issue-8-merge-models/posts/merge-models .\n'''
    (best_path/'README.md').write_text(card+'\nSelected merge: '+best+'\n\nTest results:\n```json\n'+json.dumps(verified['scores'],indent=2)+'\n```\n')
    save_results()
    import shutil
    shutil.copy2(Path(__file__), best_path/'train.py')
    if push:
        assert token, 'No HF token injected; use train_and_push with hf secret'
        import shutil
        for filename in ('results.json','train.jsonl','validation.jsonl','test.jsonl'):
            shutil.copy2(out/filename,best_path/filename)
        api=HfApi(token=token); repo='theoriclabs/qwen3-0.6b-support-order-merge'
        api.create_repo(repo,exist_ok=True)
        api.upload_folder(repo_id=repo,folder_path=str(best_path),commit_message='Publish measured support/order merge experiment')
        results['hub_repo']=repo;save_results()
    print('RESULT',json.dumps({k:v for k,v in results.items() if k not in ('models','training','merge_configs')}),flush=True)
    return {**{k:v for k,v in results.items() if k not in ('models','training','merge_configs')},
            'scores':{k:v['test']['scores'] for k,v in results['models'].items()}}


@app.function(gpu='cheap',image=image,timeout=1800)
def train(sample:bool=False,epochs:int=3)->dict:
    return _train(sample=sample,epochs=epochs)


@app.function(gpu='cheap',image=image,timeout=1800,secrets=[hf_secret])
def train_and_push(sample:bool=False,epochs:int=3)->dict:
    return _train(sample=sample,epochs=epochs,push=True)
