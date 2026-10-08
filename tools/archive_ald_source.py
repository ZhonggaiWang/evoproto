"""Archive the exact source/data-list bytes recorded by an existing ALD study.

Weights, images, environments and earlier archives remain separate. Optional
documentation and helpers record their current bytes, not training provenance.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tarfile

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('_ald_source_archive_summary',
                                             ROOT / 'tools/summarize_ald_study.py')
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)
MAX_BYTES = 16 * 1024 * 1024
AUXILIARY = (
    'README.md', 'docs/ald_gate_new_protocol.md', 'docs/ald_gate_new_results.md',
    'docs/ald_fusion_protocol.md', 'docs/ald_fusion_results.md',
    'scripts/activate_baseline_env.sh', '.runtime/requirements-locked.txt',
    '.runtime/ald-gate-policy-cuda-verification.json',
    '.runtime/ald-gate-policy-cuda-loss-gradient-verification.json',
    '.runtime/ald-gate-smoke-legacy-crossrun-comparison.json',
    '.runtime/ald-gate-dataflow-audit.json',
    '.runtime/ald-launch-process-namespace-verification.json',
    '.runtime/ald-paper-metric-scope-verification.json',
    'tools/summarize_ald_study.py', 'tools/summarize_ald_gate_study.py',
    'tools/plot_ald_diagnostics.py', 'tools/queue_ald_gate_study.py',
    'tools/monitor_ald_study.py', 'tools/archive_ald_source.py',
)


def checked(path):
    path = Path(path)
    path = summary.checked_path(path if path.is_absolute() else ROOT / path)
    if path.exists() and not (path.is_file() or path.is_dir()):
        raise ValueError(f'Not an ordinary project node: {path}')
    return path


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def regular_bytes(path, role):
    path = checked(path)
    relative = path.relative_to(ROOT)
    if any(path.is_relative_to(ROOT / '.runtime' / directory)
           for directory in ('env', 'envs', 'pkgs', 'pip', 'cache', 'tmp', 'data-prep')):
        raise ValueError(f'Environment/cache/data preparation content is excluded: {relative}')
    allowed = {'.py'} if role == 'training' else {'.txt', '.npy'} if role == 'lists' else {
        '.py', '.md', '.sh', '.json', '.txt', '.toml', '.yaml', '.yml'}
    if path.suffix not in allowed or path.name.startswith('source_snapshot'):
        raise ValueError(f'Excluded archive member type: {relative}')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > MAX_BYTES:
            raise ValueError(f'Expected one ordinary small independent file: {relative}')
        content = handle.read()
        after = os.fstat(handle.fileno())
    checked(path)
    current = path.stat()
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_nlink, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(after) != stamp(current):
        raise ValueError(f'Archive input changed during read: {relative}')
    return content


def exclusive_output(path):
    path = checked(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        checked(path)
        current, opened = path.stat(), os.fstat(descriptor)
        if opened.st_nlink != 1 or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f'Archive output identity changed: {path}')
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def archive(args):
    study = checked(args.study)
    manifest_path = checked(study / 'study.json')
    if not study.is_dir() or not manifest_path.is_file():
        raise ValueError('Study must already exist with study.json; no study directory is created')
    manifest_bytes = regular_bytes(manifest_path, 'auxiliary')
    manifest = json.loads(manifest_bytes, parse_constant=reject_constant)
    code, lists = manifest['code_sha256'], manifest['dataset_lists_sha256']
    if not isinstance(code, dict) or not code or not isinstance(lists, dict) or not lists:
        raise ValueError('Expected nonempty source and data-list fingerprints')
    output = checked(args.output_dir) if args.output_dir else study
    if args.output_dir and not output.is_relative_to(ROOT / '.runtime'):
        raise ValueError('--output-dir is restricted to the project .runtime for independent checks')
    archive_path = checked(output / 'source_snapshot.tar.gz')
    receipt_path = checked(output / 'source_snapshot.json')
    if not args.dry_run and (archive_path.exists() or receipt_path.exists()):
        raise ValueError('Existing archive/receipt is preserved; choose a fresh output directory')
    members, hashes, roles = {}, {}, {}
    def add(path, role, expected=None):
        path = checked(path)
        name = str(path.relative_to(ROOT))
        content = regular_bytes(path, role)
        digest = hashlib.sha256(content).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError(f'Recorded {role} bytes differ from manifest: {name}')
        if name in hashes and hashes[name] != digest:
            raise ValueError(f'Conflicting archive bytes: {name}')
        members[name], hashes[name], roles[name] = content, digest, role
    for name, digest in code.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Expected ordinary relative training path: {name}')
        add(ROOT / name, 'training', digest)
    for name, digest in lists.items():
        if '..' in Path(name).parts:
            raise ValueError(f'Noncanonical data-list path: {name}')
        add(ROOT / name, 'lists', digest)
    missing = []
    optional = [ROOT / name for name in AUXILIARY]
    optional.extend(sorted((ROOT / 'tests').glob('test_ald*.py')))
    for path in optional:
        if checked(path).is_file():
            if str(path.relative_to(ROOT)) not in members:
                add(path, 'auxiliary')
        else:
            missing.append(str(path.relative_to(ROOT)))
    for path in args.include:
        path = checked(path)
        if str(path.relative_to(ROOT)) not in members:
            add(path, 'auxiliary')
    manifest_name = str(manifest_path.relative_to(ROOT))
    members[manifest_name] = manifest_bytes
    hashes[manifest_name] = hashlib.sha256(manifest_bytes).hexdigest()
    roles[manifest_name] = 'auxiliary'
    counts = {role: sum(value == role for value in roles.values())
              for role in ('training', 'lists', 'auxiliary')}
    if counts['training'] != len(code) or counts['lists'] != len(lists):
        raise ValueError('Source/list paths overlap or are not unique canonical members')
    if summary.sha256(manifest_path) != hashes[manifest_name]:
        raise ValueError('Study manifest changed while preparing archive')
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'study': str(study), 'counts': counts,
                          'missing_optional_files': missing}, sort_keys=True))
        return 0
    checked(output).mkdir(parents=True, exist_ok=True)
    checked(output)
    with os.fdopen(exclusive_output(archive_path), 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as handle:
                for name in sorted(members):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(members[name]), 0o644, 0
                    handle.addfile(info, io.BytesIO(members[name]))
        raw.flush()
        os.fsync(raw.fileno())
    checked(archive_path)
    receipt = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'archive_sha256': summary.sha256(archive_path),
        'archive_bytes': archive_path.stat().st_size, 'files': hashes,
        'training_fingerprint': code, 'dataset_lists_sha256': lists,
        'study_manifest_path': str(manifest_path), 'study_manifest_sha256': hashes[manifest_name],
        'shared_step0_checkpoint_sha256': manifest.get('shared_initialization', {}).get('checkpoint_sha256'),
        'member_counts': counts, 'missing_optional_files': missing,
        'auxiliary_files': {name: digest for name, digest in hashes.items() if roles[name] == 'auxiliary'},
        'auxiliary_scope': 'Documentation and auxiliary tools are byte snapshots taken when read; '
                           'their SHA values are outside the frozen training fingerprint.',
        'scope': 'Exact recorded training source and data lists plus separately identified helpers. '
                 'Weights, image data, environments and existing archives remain separate.'}
    with os.fdopen(exclusive_output(receipt_path), 'w') as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    provenance, _ = summary.audit_sources(output, manifest, allow_drift=False)
    gate_contract = None
    if 'experiments/ald_gate_new_v1/ald_stats.py' in code:
        spec = importlib.util.spec_from_file_location('_ald_source_archive_gate_summary',
                                                     ROOT / 'tools/summarize_ald_gate_study.py')
        gate_summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate_summary)
        gate_provenance, names, counters = gate_summary.source_constants(output, manifest)
        gate_contract = {'archive_verified': gate_provenance['archive_verified'],
                         'class_count': len(names), 'counter_count': len(counters)}
    print(json.dumps({'study': str(study), 'output': str(output), 'counts': counts,
                      'archive_sha256': receipt['archive_sha256'],
                      'archive_verified': provenance['archive_verified'],
                      'gate_source_contract': gate_contract}, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output-dir', help='Fresh independent output under project .runtime')
    parser.add_argument('--include', action='append', default=[], help='Additional ordinary small helper file')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        return archive(args)
    except Exception as exc:
        print(f'Archive failed: {exc}. Existing or partial outputs are preserved.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
