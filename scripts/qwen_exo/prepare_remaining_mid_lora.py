import argparse,collections,hashlib,json,os,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2');os.environ['TOKENIZERS_PARALLELISM']='false'
from qwen_exo_booster.native_mid_lora_self_think import load_profile_tokenizer,clean_message,example_from_record,example_to_record
from qwen_exo_booster.native_mid_lora_recovery import prepare_recovered_example

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(f,obj):f.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
def main(args):
 receipt_path=Path(args.receipt);prior=receipt_path.parent;receipt=json.loads(receipt_path.read_text());source=Path(receipt['source_path']);assert sha(source)==receipt['source_sha256'];out=Path(args.output);out.mkdir(mode=0o700,parents=True,exist_ok=False)
 tokenizer,binding=load_profile_tokenizer(receipt['profile_path']);assert binding==receipt['profile_binding']
 config=json.loads((Path(receipt['profile_path'])/'config.json').read_text());cap=config.get('text_config',config)['max_position_embeddings'];groups={};splits={}
 for name in ('index.jsonl','excluded.jsonl'):
  for line in (prior/name).open():
   row=json.loads(line)
   if 'task_group' in row:
    lineid=row['source_line'];group=row['task_group'];assert lineid not in groups or groups[lineid]==group;groups[lineid]=group
    if 'split' in row:
     assert group not in splits or splits[group]==row['split'];splits[group]=row['split']
 assert all(g in splits for g in groups.values()),'Cannot infer unobserved original split'
 accepted={}
 for line in (prior/'prepared.jsonl').open():
  row=json.loads(line);accepted[row['provenance']['sample_hash']]=(row,line)
 trained=set();base=Path('/root/autodl-tmp/bench')
 ancestry=[('gym-from134-20261031T193749Z',None),('gym-decode-resume161-20261101T172106Z',None),('fixed-think-train183-20261102T150735Z',None)] if args.dataset=='gym' else [('self-think-lora-20261028T195847Z/research-v11',None),('self-think-lora-20261028T195847Z/full330-warm-v1',34)]
 for relative,last_index in ancestry:
  events = base/relative/('training/events.jsonl' if args.dataset=='gym' else 'events.jsonl')
  for line in events.open():
   row=json.loads(line)
   if row.get('event')!='training_sample' or row.get('arm')!='self_think_sft' or 'loss' not in row:continue
   if last_index is not None and row.get('sample_index',0)>last_index:continue
   trained.add(row['provenance']['sample_hash'])
 seen=set();counts=collections.Counter();exclusions=collections.Counter();start=time.time();full_lengths=[];short_lengths=[];largest=0
 with (out/'prepared.jsonl').open('x') as prepared,(out/'index.jsonl').open('x') as index,(out/'excluded.jsonl').open('x') as excluded:
  for source_line,line in enumerate(source.open(),1):
   messages=json.loads(line)['messages'];rolling=hashlib.sha256(b'[')
   for target,m in enumerate(messages):
    cleaned=clean_message(m)
    if target:rolling.update(b',')
    rolling.update(json.dumps(cleaned,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    if m.get('role')!='assistant':continue
    counts['source_targets']+=1;current=rolling.copy();current.update(b']');sample_hash=current.hexdigest()
    if sample_hash in trained:counts['already_trained_occurrences']+=1;continue
    if sample_hash in seen:counts['duplicate_untrained_occurrences']+=1;continue
    seen.add(sample_hash);counts['unique_untrained_candidates']+=1
    if sample_hash in accepted:
     record,raw=accepted[sample_hash];prepared.write(raw if raw.endswith('\n') else raw+'\n');example=example_from_record(record);counts['original_short_retained']+=1;mode='original_short'
    else:
     provenance={'source_sha256':receipt['source_sha256'],'source_line':source_line,'task_group':groups[source_line],'split':splits[groups[source_line]],'split_seed':receipt['split_seed'],'profile_binding':binding}
     try:
      example=prepare_recovered_example(tokenizer,messages,target,generation_context_tokens=cap,total_tokens=32768,max_think_tokens=receipt['max_think_tokens'],min_think_tokens=receipt['min_think_tokens'],provenance=provenance)
      record=example_to_record(example);example_from_record(record);assert example.provenance['sample_hash']==sample_hash and example.provenance['profile_binding']==binding
     except ValueError as error:
      reason=str(error)
      if len(reason)>100 or not reason.replace('_','').isalnum():reason='recovery_template_validation_failed'
      exclusions[reason]+=1;dump(excluded,{'source_line':source_line,'target_index':target,'sample_hash':sample_hash,'reason':reason});continue
     dump(prepared,record);counts['recovered']+=1;mode='recovered';counts['with_history_trim']+=bool(example.provenance['removed_message_count']);counts['with_output_hiding']+=bool(example.provenance['hidden_body_count'])
    counts['prepared']+=1;counts[example.provenance['split']]+=1
    full=len(example.generation_input_ids if example.generation_input_ids is not None else example.question_input_ids);full_lengths.append(full);short_lengths.append(len(example.question_input_ids));largest=max(largest,full)
    dump(index,{'prepared_line':counts['prepared'],'sample_hash':sample_hash,'source_line':example.provenance['source_line'],'target_index':target,'task_group':example.provenance['task_group'],'split':example.provenance['split'],'mode':mode,'generation_tokens':full,'training_prefix_tokens':len(example.question_input_ids),'think_budget':example.allowed_think_tokens})
    if counts['prepared']%25==0:
     prepared.flush();index.flush();(out/'progress.json').write_text(json.dumps({'counts':dict(counts),'exclusions':dict(exclusions),'seconds':time.time()-start}));print(json.dumps({'dataset':args.dataset,'prepared':counts['prepared'],'recovered':counts['recovered'],'max_generation_tokens':largest}),flush=True)
 assert sha(source)==receipt['source_sha256']
 summary={'dataset':args.dataset,'source':str(source),'source_sha256':receipt['source_sha256'],'original_receipt':str(receipt_path),'counts':dict(counts),'exclusions':dict(exclusions),'generation_context_limit':cap,'training_context_limit':32768,'max_generation_prefix':max(full_lengths),'max_training_prefix':max(short_lengths),'min_generation_prefix':min(full_lengths),'generation_prefix_tokens_sum':sum(full_lengths),'already_trained_unique_samples':len(trained),'split_preserved':True,'original_short_records_unchanged':True,'seconds':time.time()-start,'artifacts':{name:sha(out/name) for name in ['prepared.jsonl','index.jsonl','excluded.jsonl']},'content_displayed':False,'training_started':False}
 (out/'receipt.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--output',required=True);p.add_argument('--dataset',choices=['gym','claude'],required=True);main(p.parse_args())
