"""Concurrent supervisors must select their own immutable training packages."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def function_text(script, name, following):
    text = (REPO / 'scripts' / script).read_text()
    return text[text.index(name + '() {'):text.index(following + '() {')]


def test_concurrent_run_markers_and_default(tmp_path):
    (tmp_path / '.last_sync_hash').write_text('111111111111\n')
    first = tmp_path / 'relative.hash'
    second = tmp_path / 'absolute.hash'
    first.write_text('222222222222\n')
    second.write_text('333333333333\n')
    script = '''set -euo pipefail
ensure_workers() { :; }
ssh_do() { printf 'BOOT_OK %s\\n' "$EXPECTED"; }
log() { :; }
timeout() { shift; "$@"; }
gcloud() { printf '%s\\n' "$@"; }
'''
    script += function_text('tpu_ops.sh', 'cmd_syncstatus', 'cmd_launch')
    script += function_text('babysit.sh', 'create_qr', 'finish_run')
    script += '\ncmd_syncstatus\ncreate_qr\n'

    def run(case):
        marker, expected = case
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('FASTWAM_PINNED_CODE_PACKAGE_')
               and k != 'FASTWAM_STAGED_HASH_FILE'}
        env.update(REPO=str(REPO), WORKERS='0', TPU_TMP='/tmp/fastwam',
                   TPU='test', ZONE='test', ACC='test', RT='test', BUCKET='test',
                   CODE_PREFIX='test', TPU_USER='test', LOG='/dev/null', EXPECTED=expected)
        # The root marker lives in a temporary checkout with a startup-script stub.
        env['REPO'] = str(tmp_path)
        if marker is not None:
            env['FASTWAM_STAGED_HASH_FILE'] = str(marker)
        result = subprocess.run(['bash', '-c', script], env=env, text=True,
                                capture_output=True, check=True)
        assert f'1/1 workers report BOOT_OK {expected}' in result.stdout
        assert f'fastwam-hash={expected},' in result.stdout

    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / '_vm_startup.sh').touch()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run, [(None, '111111111111'),
                           (first, '222222222222'), (second, '333333333333')]))
    assert (tmp_path / '.last_sync_hash').read_text() == '111111111111\n'
