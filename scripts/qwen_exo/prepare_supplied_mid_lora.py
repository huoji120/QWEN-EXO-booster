"""Content-blind conversion of existing reasoning; never calls a model."""
import argparse,collections,copy,hashlib,json,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2')
from qwen_exo_booster.native_mid_lora_self_think import load_profile_tokenizer,prepare_example,example_to_record,example_from_record,_hash,_ids,load_prepared_examples,validate_prepared_profile
from qwen_exo_booster.native_mid_lora_recovery import prepare_recovered_example
from qwen_exo_booster.native_precomputed_think import make_supplied_record,validate_record,load_precomputed_thinks

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(f,x):f.write(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def main(args):
 source=Path(args.source);out=Path(args.output);out.mkdir(mode=0o700,exist_ok=False);original_sha=sha(source);tokenizer,binding=load_profile_tokenizer(args.profile);cap=json.loads((Path(args.profile)/'config.json').read_text())['text_config']['max_position_embeddings'];counts=collections.Counter();failures=[];seen=set();source_keys=[]
 with (out/'prepared.jsonl').open('x') as prepared,(out/'source-think-index.jsonl').open('x') as thoughts:
  for line_no,line in enumerate(source.open(encoding='utf-8-sig'),1):
   row=json.loads(line);counts['rollouts']+=1;messages=copy.deepcopy(row['messages']);source_keys.append(_hash(row['task_id']))
   for msg in messages:
    if msg.get('content') is None:msg['content']=''
    for call in msg.get('tool_calls',[]):
     fn=call['function'];a=fn['arguments']
     if isinstance(a,str):a=json.loads(a);assert isinstance(a,dict);fn['arguments']=a;counts['argument_objects_normalized']+=1
   for target,m in enumerate(messages):
    if m['role']!='assistant':continue
    counts['assistant_targets']+=1;reason=m.get('reasoning_content') or '';assert isinstance(reason,str)
    # Native framing, not generated or invented reasoning. Empty source stays empty.
    tokens=_ids(tokenizer,reason+'\n</think>');budget=len(tokens)
    provenance={'source_sha256':original_sha,'source_line':line_no,'task_group':_hash(row['task_id']),'split':'train','split_seed':'provided_same_task_train_only','profile_binding':binding,'source_reasoning_sha256':hashlib.sha256(reason.encode()).hexdigest(),'source_reasoning_present':bool(reason)}
    provenance['supplied_supervision'] = 'think_and_answer' if args.supervise_reasoning else 'answer_only'
    try:
     try:example=prepare_example(tokenizer,messages,target,max_think_tokens=budget,min_think_tokens=budget,provenance=provenance)
     except ValueError as e:
      if str(e) not in ('complete_prefix_insufficient_think_budget','ordinary_complete_prefix_overflow'):raise
      example=prepare_recovered_example(tokenizer,messages,target,generation_context_tokens=cap,max_think_tokens=budget,min_think_tokens=budget,provenance=provenance)
     sample=example.provenance['sample_hash']
     if sample in seen:counts['duplicate_targets']+=1;continue
     seen.add(sample);example_from_record(example_to_record(example));counts['accepted']+=1;counts['with_supplied_think' if reason else 'without_source_think']+=1
     counts['recovered_train_context']+=example.generation_input_ids is not None
     dump(prepared,example_to_record(example));dump(thoughts,{'sample_hash':sample,'token_ids':list(tokens),'source_reasoning_sha256':provenance['source_reasoning_sha256'],'reasoning_present':bool(reason)})
    except ValueError as e:
     code=str(e)
     if len(code)>100 or not code.replace('_','').isalnum():code='source_template_invalid'
     failures.append({'source_line':line_no,'target_index':target,'reason':code})
 source_hash=sha(out/'prepared.jsonl');lengths=[]
 with (out/'think-cache.jsonl').open('x') as cached,(out/'train-ready.jsonl').open('x') as ready,(out/'source-think-index.jsonl').open() as thoughts:
  for example in load_prepared_examples(out/'prepared.jsonl'):
   t=json.loads(next(thoughts));assert t['sample_hash']==example.provenance['sample_hash']
   rec=make_supplied_record(example,t['token_ids'],prepared_data_sha256=source_hash,source_sha256=original_sha,source_reasoning_sha256=t['source_reasoning_sha256'],reasoning_present=t['reasoning_present'],supervision=example.provenance['supplied_supervision']);assembled=validate_record(rec,example,prepared_data_sha256=source_hash)
   start=len(example.question_input_ids);end=start+len(t['token_ids']);assert all(v==-100 for v in assembled['labels'][:start])
   if args.supervise_reasoning and t['reasoning_present']:assert assembled['labels'][start:end]==t['token_ids']
   else:assert all(v==-100 for v in assembled['labels'][start:end])
   assert len(assembled['input_ids'])<=32768;lengths.append(len(assembled['input_ids']));dump(cached,rec);dump(ready,assembled)
  assert next(thoughts,None) is None
 examples=load_prepared_examples(out/'prepared.jsonl');validate_prepared_profile(examples,args.profile);load_precomputed_thinks(out/'think-cache.jsonl',examples,prepared_data_sha256=source_hash)
 assert sha(source)==original_sha
 report={'source_sha256':original_sha,'counts':dict(counts),'unique_tasks':len(set(source_keys)),'heldout':0,'heldout_reason':'same_task_rollouts_not_independent_holdout','reasoning_policy':'provided_source_reasoning_'+('think_and_answer' if args.supervise_reasoning else 'answer_only'),'no_generation':True,'max_tokens':max(lengths),'min_tokens':min(lengths),'exclusions':failures,'artifacts':{n:sha(out/n) for n in ['prepared.jsonl','think-cache.jsonl','train-ready.jsonl']},'training_started':False}
 (out/'receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='exclusions'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--profile',required=True);p.add_argument('--supervise-reasoning',action='store_true');main(p.parse_args())
