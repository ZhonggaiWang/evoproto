from explore import *
import zipfile,subprocess
analysis=json.loads((U/'analysis.json').read_text())
text=safe_path(U/'result.txt').read_text()
text+='\n补充：边界修正相对于同一空间设置，对全部21类均为正收益；但整个新协议相对于原448设置，cat约下降3.069、dog约下降5.153、pottedplant约下降0.207，详见逐类记录。\n'
safe_path(U/'result.txt').write_text(text,encoding='utf-8')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert not processes.strip(),processes
atomic_json(U/'progress.json',dict(status='complete',target=71.0,achieved=analysis['achieved'],new_training_updates=0,all_gpus_released=True,gpu_snapshot=gpu,four_card_contacted=False,utc=now()))
audit=json.loads((U/'completion_audit.json').read_text());audit.update(all_gpus_released=True,gpu_snapshot=gpu,cli_smoke_passed=True,refinement_improves_all21_classes=all(v>0 for v in analysis['refinement_class_delta'].values()))
atomic_json(U/'completion_audit.json',audit)
atomic_json(U/'next_goal_state.json',dict(status='complete',target_miou=71.0,best_miou=analysis['achieved'],recommended_deployment=str(U/'deployment.json'),new_training_updates=0,original_protocol_miou=70.04202315102656,main_remaining_errors='cat/dog regressions from changed spatial scale; small-object refinement tradeoff. Further work needs a new user request, not automatic continuation.',utc=now()))
bundle=safe_path(U/'implementation_and_records.zip')
paths=sorted(set(list(E.glob('*.py'))+list(S.rglob('*.py'))+[p for p in U.rglob('*') if p.is_file() and p.suffix in ['.json','.txt','.log','.png'] and p.name!='manifest.json']))
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
 for p in paths:
  safe_path(p);z.write(p,str(p.relative_to(R)))
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
names=['result.txt','analysis.json','deployment.json','method_provenance.json','completion_audit.json','progress.json','next_goal_state.json','implementation_and_records.zip']
atomic_json(U/'manifest.json',dict(files={n:dict(bytes=(U/n).stat().st_size,sha256=digest(U/n)) for n in names},archive_files=len(paths),weights_included=False,checkpoint=json.loads((U/'deployment.json').read_text())['checkpoint'],utc=now()))
print(json.dumps(dict(files=len(paths),archive_bytes=bundle.stat().st_size,gpus=gpu,achieved=analysis['achieved'])))
