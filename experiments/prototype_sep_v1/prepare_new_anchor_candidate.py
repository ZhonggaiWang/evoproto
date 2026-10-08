"""Prepare one isolated new-source selector candidate from frozen B."""
from pathlib import Path
import ast
import json
import shutil
import sys

R = Path('/ML-vePFS/infra_rd/kun/others/wzg/workspace/evoproto')
E = R / 'experiments/prototype_sep_v1'
B = E / 'b_semantic/src'
C = E / 'c_new_anchor/src'
sys.path.insert(0, str(B))
from kd_runtime import safe_path, atomic_json, digest, now

def hashes(src):
    for path in src.rglob('*'):
        safe_path(path)
    return {str(p.relative_to(src)): digest(p) for p in sorted(src.rglob('*.py'))
            if '__pycache__' not in p.parts}

study = json.loads(safe_path(R / 'runs/prototype_sep_v1/formal/b_semantic/study.json').read_text())
preflight = json.loads(safe_path(E / 'semantic_preflight.json').read_text())
parent = hashes(B)
assert parent == study['source_sha256'] == preflight['source_sha256']
assert preflight['passed'] is True
safe_path(C)
assert not C.exists(), 'Refuse duplicate candidate preparation'
shutil.copytree(B, C, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
shutil.copyfile(safe_path(E / 'new_anchor_geometry_selector.py'),
                safe_path(C / 'model/geometry_pair_selector.py'))
code = hashes(C)
for path in C.rglob('*.py'):
    ast.parse(path.read_text(), filename=str(path))
changed = sorted(key for key in code if code[key] != parent.get(key))
assert changed == ['model/geometry_pair_selector.py'], changed
assert hashes(B) == parent
atomic_json(E / 'new_anchor_origin.json', {
    'created_utc': now(), 'parent_source': str(B), 'parent_source_sha256': parent,
    'candidate_source': str(C), 'source_sha256': code, 'candidate_source_sha256': code,
    'changed_files': changed, 'selector_schema': 4,
    'source_anchor_policy': 'current_new_only', 'source_domain': 'new_foreground_rows_only',
    'parent_source_unchanged': True, 'KD_byte_identical': True,
    'semantic_guard_byte_identical': True,
    'scope': 'Prepared only; no training launch. Development GT diagnostics guide source policy, never training labels or precise loss weights.'})
print(json.dumps({'prepared': str(C), 'changed_files': changed,
                  'selector_sha256': code['model/geometry_pair_selector.py']}))
