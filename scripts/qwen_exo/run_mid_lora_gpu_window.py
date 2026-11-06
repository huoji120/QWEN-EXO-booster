"""Detached, explicitly authorized GPU window with independent service recovery.

The plan contains trusted operator argv arrays, never commands from training data.
This controller changes no service configuration, packages, model or adapter binding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def identity(pid):
    try:
        # Fields after comm's closing parenthesis start at process state (field 3).
        return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def alive(pid, started):
    return bool(pid and started and identity(pid) == started)


def health(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except Exception:
        return 0


def terminate(pid, started):
    if not alive(pid, started):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while alive(pid, started) and time.monotonic() < deadline:
            time.sleep(1)
        if alive(pid, started):
            os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def restore(plan):
    result = subprocess.run(plan['supervisor'] + ['start', 'qwen'], capture_output=True, text=True, timeout=180)
    deadline = time.monotonic() + plan.get('recovery_seconds', 1200)
    while time.monotonic() < deadline:
        status = health(plan['health_url'])
        if status == 200:
            return {'health': 200, 'start_exit': result.returncode, 'restored': True}
        time.sleep(3)
    return {'health': health(plan['health_url']), 'start_exit': result.returncode, 'restored': False}


def watch(state_path):
    state_path = Path(state_path)
    while True:
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError):
            time.sleep(2); continue
        if state.get('phase') == 'finished':
            return
        expired = time.time() > state['deadline']
        if not alive(state['pid'], state['started']) or expired:
            if expired and alive(state['pid'], state['started']):
                os.kill(state['pid'], signal.SIGTERM)
            terminate(state.get('child_pid'), state.get('child_started'))
            plan = json.loads(Path(state['plan']).read_text())
            recovery = restore(plan) if state.get('stop_requested') else {'restored': True, 'service_untouched': True}
            save(state_path.parent / 'watchdog-recovery.json', {'reason': 'deadline' if expired else 'controller_exit', **recovery})
            return
        time.sleep(2)


def run(plan_path, authorize_stop):
    if not authorize_stop:
        raise PermissionError('Stopping service requires explicit --authorize-stop')
    plan_path = Path(plan_path).resolve(); plan = json.loads(plan_path.read_text())
    out = Path(plan['output']).resolve(); out.mkdir(parents=True, exist_ok=False)
    if not plan['stages'] or any(not stage.get('argv') for stage in plan['stages']):
        raise ValueError('Empty execution plan')
    if health(plan['health_url']) != 200:
        raise RuntimeError('Original service is not healthy; refusing to acquire GPU window')
    config_hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in plan['immutable_config_files']}
    state_path = out / 'state.json'
    state = {'pid': os.getpid(), 'started': identity(os.getpid()), 'plan': str(plan_path),
             'deadline': time.time() + plan['window_seconds'], 'phase': 'armed', 'stop_requested': False,
             'stages': [], 'baseline_config_sha256': config_hashes}
    save(state_path, state)
    with (out / 'watchdog.log').open('ab') as log:
        watchdog = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watch', str(state_path)],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('GPU window interrupted by signal %d' % signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    child = None
    try:
        state.update(phase='stopping_service', stop_requested=True); save(state_path, state)
        subprocess.run(plan['supervisor'] + ['stop', 'qwen'], check=True, timeout=180)
        deadline = time.monotonic() + 90
        while True:
            query = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True, timeout=30)
            free = int(query.stdout.strip().split('\n')[0])
            if free >= plan['minimum_free_mib']:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU was not released by the owned service')
            time.sleep(2)
        state['free_mib_after_stop'] = free
        for stage in plan['stages']:
            state.update(phase=stage['name'], child_pid=None, child_started=None); save(state_path, state)
            env = dict(os.environ, **plan.get('environment', {}))
            env.pop('CUDA_VISIBLE_DEVICES', None)
            begin = time.time()
            with (out / (stage['name'] + '.log')).open('ab') as log:
                child = subprocess.Popen(stage['argv'], env=env, cwd=plan['source'],
                    start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                state.update(child_pid=child.pid, child_started=identity(child.pid)); save(state_path, state)
                try:
                    code = child.wait(timeout=stage['timeout_seconds'])
                except subprocess.TimeoutExpired:
                    terminate(child.pid, state['child_started']); raise
            state['stages'].append({'name': stage['name'], 'exit_code': code, 'seconds': time.time() - begin})
            state.update(child_pid=None, child_started=None); save(state_path, state)
            if code:
                raise RuntimeError('Stage failed: %s (exit %d)' % (stage['name'], code))
        state['status'] = 'completed'
    except BaseException as error:
        state['status'] = 'failed'; state['error'] = '%s: %s' % (type(error).__name__, error)
    finally:
        if child is not None and child.poll() is None:
            terminate(child.pid, state.get('child_started'))
        state.update(phase='restoring_service', child_pid=None, child_started=None); save(state_path, state)
        try:
            state['recovery'] = restore(plan)
            state['config_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in config_hashes.items())
            if not state['recovery']['restored'] or not state['config_unchanged']:
                state['status'] = 'failed'
        except BaseException as error:
            state['recovery'] = {'restored': False, 'error': '%s: %s' % (type(error).__name__, error)}
            state['status'] = 'failed'
        if state['recovery'].get('restored'):
            state['phase'] = 'finished'
        # On recovery failure leave the watchdog armed: it retries after this owner exits.
        save(state_path, state)
    print(json.dumps({'status': state['status'], 'recovery': state['recovery'], 'state': str(state_path)}), flush=True)
    return 0 if state['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan'); parser.add_argument('--authorize-stop', action='store_true'); parser.add_argument('--watch')
    args = parser.parse_args()
    if args.watch:
        watch(args.watch)
    elif args.plan:
        sys.exit(run(args.plan, args.authorize_stop))
    else:
        parser.error('--plan or --watch is required')
