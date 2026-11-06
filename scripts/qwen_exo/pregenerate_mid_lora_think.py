"""Serial, content-blind native SGLang Think capture. Never starts training."""
from __future__ import annotations
import argparse,hashlib,json,os,time,urllib.request,urllib.error,uuid
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')

def _same_file(left, right):
 try:
  return os.path.samefile(left, right)
 except (FileNotFoundError, OSError):
  return False

def _reject_source_artifact_collisions(source, root):
 source.stat()
 candidates=[root]
 if root.is_dir():
  for directory,_,files in os.walk(root,followlinks=False):
   candidates.extend(Path(directory)/name for name in files)
 for candidate in candidates:
  if _same_file(source,candidate):
   raise ValueError('source_collides_with_output_artifact')

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic(path,value):
 temp=path.with_suffix(path.suffix+'.partial')
 with temp.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def call(base,path,body=None,token=None,method=None):
 headers={'Content-Type':'application/json'}
 if token:headers['Authorization']='Bearer '+token
 request=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),headers=headers,method=method)
 with urllib.request.urlopen(request,timeout=1800) as response:return json.load(response)

def execute(args):
 from transformers import AutoTokenizer
 from types import SimpleNamespace
 from qwen_exo_booster.native_mid_lora_self_think import load_prepared_examples,example_to_record,validate_prepared_profile,generation_plan
 from qwen_exo_booster.native_precomputed_think import make_record,validate_record
 from qwen_exo_booster.think_shaping_policy import suppress_token_ids,think_shaping_params
 source=Path(args.source).resolve();root=Path(args.output).resolve();_reject_source_artifact_collisions(source,root);root.mkdir(parents=True,exist_ok=True);root.chmod(0o700)
 records=root/'records';records.mkdir(exist_ok=True)
 # Process-wide exclusive lock survives neither normal exit nor crash.
 import fcntl
 with (root/'worker.lock').open('a') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  key=None;manifest=None;started=time.time();counts={'accepted':0,'rejected':0,'pending':0};current=None
  try:
   data_sha=sha(source);examples=load_prepared_examples(source);validate_prepared_profile(examples,args.base_profile)
   issued=call(args.base_url,'/qwen-exo/api-keys',{'label':'serial-think-'+uuid.uuid4().hex[:12],'memory_scope':'private_only'});key=issued['id'];token=issued['token']
   runtime_limits = {}
   def identity():
    info=call(args.base_url,'/server_info',token=token);model=call(args.base_url,'/model_info',token=token);profile=Path(model['model_path']);cfg=json.loads((profile/'config.json').read_text());adapter=cfg['qwen_exo_native_mid_lora'];catalog=json.loads(Path(args.catalog).read_text());fp=catalog['active_model_fingerprint']
    reserved = max(int(info.get('speculative_eagle_topk') or 0)*int(info.get('speculative_num_steps') or 0),int(info.get('max_speculative_num_draft_tokens') or info.get('speculative_num_draft_tokens') or 0)) if info.get('speculative_algorithm') else 0
    observed_limits = {'context_length':int(info['context_length']),'reserved_tokens':reserved}
    if runtime_limits and runtime_limits != observed_limits:raise ValueError('teacher_context_limits_changed')
    runtime_limits.update(observed_limits)
    if not all(catalog[k]==fp for k in ('applied_model_fingerprint','healthy_model_fingerprint')):raise ValueError('teacher_catalog_not_ready')
    if sha(adapter['adapter_path'])!=adapter['adapter_sha256']:raise ValueError('teacher_adapter_hash_mismatch')
    for name in ('tokenizer.json','tokenizer_config.json','chat_template.jinja'):
     if sha(profile/name)!=sha(Path(args.base_profile)/name):raise ValueError('teacher_tokenizer_mismatch')
    shaping={k.removeprefix('qwen_exo_'):info[k] for k in ('qwen_exo_think_suppress_after_tokens','qwen_exo_think_suppress_strength','qwen_exo_think_landing_start_ratio','qwen_exo_think_landing_bias')}
    return {'teacher_fingerprint':fp,'teacher_adapter_sha256':adapter['adapter_sha256'],'teacher_profile_path':str(profile)},shaping
   teacher,settings=identity();tokenizer=AutoTokenizer.from_pretrained(args.base_profile,local_files_only=True,trust_remote_code=False);suppress=suppress_token_ids(tokenizer)
   manifest={'schema':'native-serial-think-run-v1','source':str(source),'source_sha256':data_sha,'teacher':teacher,'shaping_settings':settings,'concurrency':1,'sampling_temperature':0,'sample_count':len(examples),'generation_policy':'full_prepared_causal_prefix_no_gold','training_policy':'fixed_precomputed_teacher_no_sampling_gradient'}
   mp=root/'manifest.json'
   if mp.exists():
    if json.loads(mp.read_text())!=manifest:raise ValueError('resume_manifest_mismatch')
   else:atomic(mp,manifest)
   invoked=0
   for index,example in enumerate(examples):
    sample=example.provenance['sample_hash'];current=sample;path=records/(sample+'.json')
    if path.exists():
     saved=json.loads(path.read_text())
     if saved['status']=='accepted':validate_record(saved['record'],example,prepared_data_sha256=data_sha);counts['accepted']+=1
     elif saved['status']=='rejected':counts['rejected']+=1
     else:raise ValueError('unknown_saved_sample_status')
     continue
    if (root/'stop.json').exists():break
    if args.limit is not None and invoked>=args.limit:break
    if identity()!=(teacher,settings):raise ValueError('teacher_or_shaping_changed')
    generation_ids = list(generation_plan(example).question_input_ids)
    if example.generation_context_tokens is not None and len(generation_ids)+example.allowed_think_tokens>example.generation_context_tokens:raise ValueError('full_generation_context_overflow')
    effective_budget = min(example.allowed_think_tokens,runtime_limits['context_length']-runtime_limits['reserved_tokens']-len(generation_ids))
    if effective_budget<len(example.think_termination_ids):raise ValueError('full_generation_prefix_exceeds_serving_capacity')
    if effective_budget!=example.allowed_think_tokens:
     with (root/'budget-adjustments.jsonl').open('a') as audit:audit.write(json.dumps({'sample_hash':sample,'requested':example.allowed_think_tokens,'effective':effective_budget,**runtime_limits,'full_prefix_unchanged':True})+'\n')
    shape=think_shaping_params(config=SimpleNamespace(**settings),end_token_id=example.think_close_ids[0],suppress_ids=suppress,max_reasoning_tokens=effective_budget)
    sampling={'temperature':0,'top_p':1.0,'top_k':1,'max_new_tokens':effective_budget,'stop_token_ids':list(example.think_close_ids),'skip_special_tokens':False,'no_stop_trim':True,'custom_params':{'qwen_exo_think_shaping':shape}}
    atomic(root/'status.json',{'phase':'generating','sample_index':index,'sample_hash':sample,'counts':counts,'epoch':time.time(),'pid':os.getpid(),'concurrency':1})
    begin=time.monotonic();response=call(args.base_url,'/generate',{'input_ids':generation_ids,'sampling_params':sampling,'require_reasoning':True,'return_logprob':True,'logprob_start_len':-1,'return_prompt_token_ids':True,'no_logs':True,'extra_key':'fixed-think-'+teacher['teacher_fingerprint']},token)
    seconds=time.monotonic()-begin;invoked+=1
    if identity()!=(teacher,settings):raise ValueError('teacher_changed_during_generation')
    meta=response['meta_info'];tokens=response['output_ids']
    if response.get('prompt_token_ids')!=generation_ids:raise ValueError('actual_generation_prefix_mismatch')
    if tokens!=[row[1] for row in meta['output_token_logprobs']] or len(tokens)!=meta['completion_tokens']:raise ValueError('output_token_capture_mismatch')
    try:
     if meta.get('finish_reason',{}).get('type')!='stop':raise ValueError('generation_did_not_stop_normally')
     record=make_record(example,tokens,seconds=seconds,prepared_data_sha256=data_sha,teacher=teacher,shaping=shape,sampling=sampling)
    except ValueError as error:
     # No fabricated closer or answer: persist rejected raw token evidence privately.
     reason=str(error)
     if not reason.replace('_','').isalnum() or len(reason)>100:reason='think_validation_rejected'
     atomic(path,{'status':'rejected','sample_hash':sample,'reason':reason,'token_ids':tokens,'seconds':seconds,'finish_type':meta.get('finish_reason',{}).get('type'),'teacher':teacher,'prepared_data_sha256':data_sha});counts['rejected']+=1
    else:atomic(path,{'status':'accepted','record':record});counts['accepted']+=1
    with (root/'progress.jsonl').open('a') as f:f.write(json.dumps({'epoch':time.time(),'sample_index':index,'sample_hash':sample,'seconds':seconds,'generated_tokens':len(tokens),'status':json.loads(path.read_text())['status'],'cached_tokens':meta.get('cached_tokens')})+'\n')
   if sha(source)!=data_sha:raise ValueError('source_changed')
   # Deterministic accepted subset, plus review inventory; never discard source rows.
   split_counts={};lengths=[];count=0;rejected=[];pending=[]
   files={name:(root/(name+'.partial')).open('w') for name in ('prepared.jsonl','think-cache.jsonl','train-ready.jsonl')}
   try:
    for index,example in enumerate(examples):
     path=records/(example.provenance['sample_hash']+'.json')
     if not path.exists():pending.append(index);continue
     saved=json.loads(path.read_text())
     if saved['status']!='accepted':rejected.append({'source_index':index,'sample_hash':example.provenance['sample_hash'],'reason':saved['reason']});continue
     record=saved['record'];assembled=validate_record(record,example,prepared_data_sha256=data_sha)
     for name,value in [('prepared.jsonl',example_to_record(example)),('think-cache.jsonl',record),('train-ready.jsonl',assembled)]:files[name].write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
     split=example.provenance['split'];split_counts[split]=split_counts.get(split,0)+1;lengths.append(len(assembled['input_ids']));count+=1
    for f in files.values():f.flush();os.fsync(f.fileno())
   finally:
    for f in files.values():f.close()
   for name in files:os.replace(root/(name+'.partial'),root/name)
   atomic(root/'review.json',{'rejected':rejected,'pending_source_indices':pending})
   receipt={'status':'completed' if not pending else 'partial','input_samples':len(examples),'accepted':count,'rejected':len(rejected),'pending':len(pending),'splits':split_counts,'min_tokens':min(lengths) if lengths else None,'max_tokens':max(lengths) if lengths else None,'source_sha256':data_sha,'teacher':teacher,'seconds_this_invocation':time.time()-started,'concurrency':1,'shaping_settings':settings,'artifacts':{n:sha(root/n) for n in files},'training_started':False,'service_restarted':False,'content_displayed':False}
   atomic(root/'receipt.json',receipt);atomic(root/'status.json',{'phase':receipt['status'],'epoch':time.time(),'pid':os.getpid(),'accepted':count,'rejected':len(rejected),'pending':len(pending)});print(json.dumps(receipt),flush=True)
  except BaseException as error:
   atomic(root/'status.json',{'phase':'failed','epoch':time.time(),'pid':os.getpid(),'error_type':type(error).__name__,'error_sha256':hashlib.sha256(str(error).encode()).hexdigest(),'sample_hash':current,'counts':counts})
   raise RuntimeError(type(error).__name__+' during pre-generation; details hidden') from None
  finally:
   if key:
    call(args.base_url,'/qwen-exo/api-keys/'+key,method='DELETE')
    atomic(root/'key-revocation.json',{'revoked':True,'epoch':time.time()})

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--base-profile',required=True);p.add_argument('--base-url',default='http://127.0.0.1:30000');p.add_argument('--catalog',default='/root/autodl-tmp/qwen-exo-runtime-flash-next-existing-ple/model-catalog.json');p.add_argument('--limit',type=int);a=p.parse_args();execute(a)
