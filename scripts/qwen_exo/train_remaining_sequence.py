import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def atomic(path,value):
 temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def latest_pair(run):
 checkpoint=None
 for raw in (run/'training/events.jsonl').open():
  if raw.endswith('\n'):
   r=json.loads(raw)
   if r.get('event')=='adapter_checkpoint':checkpoint=r
 if checkpoint is None:raise RuntimeError('No committed checkpoint')
 state=checkpoint['training_state']
 if sha(checkpoint['path'])!=state['adapter_sha256'] or sha(state['path'])!=state['sha256']:raise RuntimeError('Checkpoint pair mismatch')
 import torch
 saved=torch.load(state['path'],map_location='cpu',weights_only=True)
 if saved['adapter_sha256']!=state['adapter_sha256'] or saved['progress']['optimizer_step']!=checkpoint['optimizer_step']:raise RuntimeError('Optimizer checkpoint binding mismatch')
 if len(saved['optimizer']['state'])!=32:raise RuntimeError('Optimizer parameter state count mismatch')
 for value in saved['optimizer']['state'].values():
  if int(value['step'])!=saved['progress']['optimizer_session_step'] or not torch.isfinite(value['exp_avg']).all() or not torch.isfinite(value['exp_avg_sq']).all():raise RuntimeError('Invalid optimizer state')
 return checkpoint

def build_argv(config,dataset,previous):
 from qwen_exo_booster.native_mid_lora_self_think import load_prepared_examples,file_sha256
 from qwen_exo_booster.native_precomputed_think import load_precomputed_thinks
 root=Path(config['root']);definition=config['datasets'][dataset];ready=Path(definition['output']);source=Path(definition['source']);receipt=json.loads((ready/'receipt.json').read_text())
 if receipt['status']!='completed' or receipt['pending']:raise RuntimeError('Dataset preparation incomplete')
 if sha(source)!=receipt['source_sha256']:raise RuntimeError('Prepared source changed')
 verified=json.loads((root/(dataset+'-verified.json')).read_text())
 if not verified['all_source_prefixes_rebuilt'] or not verified['all_targets_and_masks_verified']:raise RuntimeError('Dataset not independently verified')
 for name,digest in verified['artifact_hashes'].items():
  if sha(ready/name)!=digest:raise RuntimeError('Verified dataset artifact changed')
 examples=load_prepared_examples(ready/'prepared.jsonl');load_precomputed_thinks(ready/'think-cache.jsonl',examples,prepared_data_sha256=receipt['source_sha256'])
 count=sum(x.provenance['split']=='train' for x in examples)
 if count<1:raise RuntimeError('No train samples')
 checkpoint=latest_pair(Path(previous));args=list(config['base_argv']);args[1]=str(root/'source/scripts/qwen_exo/run_mid_lora_experiment.py')
 values={'--data':ready/'prepared.jsonl','--output':root/(dataset+'-training')/'training','--initial-adapter':checkpoint['path'],'--resume-training-state':checkpoint['training_state']['path'],'--initial-optimizer-steps':checkpoint['optimizer_step'],'--start-sample-index':0,'--steps':count,'--precomputed-think':ready/'think-cache.jsonl','--precomputed-think-source':source}
 if 'heldout_limit' in definition:values['--heldout-limit']=definition['heldout_limit']
 for flag,value in values.items():
  if flag in args:args[args.index(flag)+1]=str(value)
  else:args +=[flag,str(value)]
 if '--transfer-dataset' not in args:args.append('--transfer-dataset')
 return args,{'dataset':dataset,'initial_step':checkpoint['optimizer_step'],'train_samples':count,'source_sha256':receipt['source_sha256'],'cache_sha256':file_sha256(ready/'think-cache.jsonl')}

def main(args):
 root=Path(args.root);config=json.loads((root/'training-sequence.json').read_text());previous=config['initial_run']
 order=config['order']
 if not order or len(set(order))!=len(order) or any(name not in config['datasets'] for name in order):raise ValueError('Invalid dataset sequence')
 if not args.authorize_training:
  argv,summary=build_argv(config,order[0],previous);(root/(order[0]+'-training-argv.json')).write_text(json.dumps(argv,indent=2));print(json.dumps({'status':'prepared_not_started',**summary}));return
 for dataset in order:
  if (root/'stop-training.json').exists():return
  argv,summary=build_argv(config,dataset,previous);output=root/(dataset+'-training');output.mkdir(mode=0o700)
  (output/'input-binding.json').write_text(json.dumps(summary,indent=2))
  with (output/'trainer.log').open('ab') as log:
   child=subprocess.Popen(argv,env=dict(os.environ,**config['environment']),stdout=log,stderr=subprocess.STDOUT)
   start=(Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
   atomic(root/'sequence-status.json',{'phase':'training','dataset':dataset,'run':str(output),'pid':child.pid,'start':start,'epoch':time.time(),**summary})
   code=child.wait()
  if code:
   atomic(root/'sequence-status.json',{'phase':'failed','dataset':dataset,'exit_code':code,'epoch':time.time()});raise RuntimeError('Training failed; sequence stopped: '+dataset)
  checkpoint=latest_pair(output);atomic(output/'completed.json',{'final_step':checkpoint['optimizer_step'],'checkpoint':checkpoint['path']});previous=str(output)
  atomic(root/'sequence-status.json',{'phase':'dataset_completed','dataset':dataset,'run':str(output),'final_step':checkpoint['optimizer_step'],'epoch':time.time()})
 atomic(root/'sequence-status.json',{'phase':'completed','dataset':order[-1],'final_run':previous,'epoch':time.time()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--authorize-training',action='store_true');main(p.parse_args())
