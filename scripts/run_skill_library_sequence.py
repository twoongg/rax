#!/usr/bin/env python3
"""Run tasks one after another, growing a controller library between them.

For each task: prepare a session with the library (--skill-library), run it headless to the
end of its formal batch, then add the code it was evaluated with to the library
(services/skill_library.py publish), whatever its score. The next task starts from the
grown library. Design: docs/superpowers/specs/2026-11-25-skill-library-design.md.

    runtime/envs/robodojo/bin/python scripts/run_skill_library_sequence.py \\
        --library runtime/skill-library/deepseek-a \\
        --tasks pour_liquid_into_cup,press_by_number,swap_t,pour_balls_into_vase,insert_tubes,insert_key \\
        --configure-args "--sim-gpu 0 --agent-cli claude --agent-provider custom --episodes 16 --formal-episodes 10" \\
        --launch-args "--claude-base-url http://127.0.0.1:8000 --claude-key-file <key file> --claude-model <model>"

The library is created from the seed (skill_library_seed/robodojo) on first use, as a git
repository. Each finished task appends a row to --records (default <library>-runs.jsonl); a
rerun skips tasks already recorded as finished, so a stopped sequence resumes where it was.
--no-library runs the same tasks without a library, for comparison (needs --records).
"""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from services import skill_library  # noqa: E402

FINISHED = ('published', 'not-published', 'evaluated')  # Rows a rerun does not repeat.


def read_records(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_record(path, row):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, default=str) + '\n')


def finished_tasks(records):
    return {row['task'] for row in records if row.get('status') in FINISHED}


def run_one(task, *, library, prepare, launch, publish=True, min_success=None, log=print):
    """One task: prepare -> launch (exploration and formal batch) -> publish. Returns its record row."""
    row = {'task': task, 'started': time.time(), 'library': str(library) if library else None,
           'library_before': skill_library.head(library) if library else None}
    root = Path(prepare(task))
    row['session'] = str(root)
    log(f'[{task}] session {root}')
    row['returncode'] = launch(root)
    outcome = skill_library.formal_outcome(root)
    row['formal'] = f'{outcome[0]}/{outcome[1]}' if outcome else None
    if not (root/'formal.json').is_file():
        row['status'] = 'failed'
        row['reason'] = 'the session wrote no formal.json'
    elif not publish:
        row['status'] = 'evaluated'
    else:
        result = skill_library.publish(root, library, min_success=min_success)
        row['publish'] = result
        row['status'] = 'published' if result['status'] == 'published' else 'not-published'
        row['library_after'] = skill_library.head(library)
    row['finished'] = time.time()
    log(f'[{task}] {row["status"]}: formal {row["formal"]}'
        + (f', entry {row["publish"].get("slug")}' if row.get('publish', {}).get('slug') else '')
        + (f' ({row.get("reason") or row.get("publish", {}).get("reason")})'
           if row['status'] in ('failed', 'not-published') else ''))
    return row


def sequence(tasks, *, library, records, prepare, launch, publish=True, min_success=None, log=print):
    """Run the tasks in order; stop at a session that wrote no formal.json (fix it and rerun)."""
    done = finished_tasks(read_records(records))
    rows = []
    for task in tasks:
        if task in done:
            log(f'[{task}] already finished in {records}; skipped')
            continue
        row = run_one(task, library=library, prepare=prepare, launch=launch, publish=publish,
                      min_success=min_success, log=log)
        append_record(records, row)
        rows.append(row)
        if row['status'] == 'failed':
            log(f'[{task}] stopping the sequence: {row["reason"]}; rerun to retry from this task')
            break
    return rows


def configure_runner(args):
    """Prepare with configure_auto_research.py and run the launcher headless."""
    python = sys.executable

    def prepare(task):
        command = [python, str(PROJECT/'scripts/configure_auto_research.py'), '--task', task,
                   *(['--skill-library', str(args.library)] if not args.no_library else []),
                   *shlex.split(args.configure_args)]
        output = subprocess.run(command, cwd=PROJECT, check=True, capture_output=True, text=True).stdout
        config = Path(output.strip().splitlines()[-1])
        if not config.is_file():
            raise RuntimeError(f'configure printed no configuration path: {output!r}')
        return config.parent

    def launch(root):
        command = ['bash', str(PROJECT/'scripts/start_auto_research_agent.sh'), '--config',
                   str(root/'operator-input.json'), '--non-interactive', *shlex.split(args.launch_args)]
        with (root/'agent.log').open('a') as log:
            return subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT).returncode

    return prepare, launch


def deepseek_runner(args):
    """Prepare and run each session with run_deepseek_experiment.py (local vLLM endpoints, DSH)."""
    sessions = {}

    def prepare(task):
        record = Path(args.records).with_suffix('')/f'{task}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json'
        command = [sys.executable, str(PROJECT/'scripts/run_deepseek_experiment.py'), '--task', task,
                   '--record', str(record),
                   *(['--skill-library', str(args.library)] if not args.no_library else []),
                   *shlex.split(args.runner_args)]
        child = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, text=True)
        first = child.stdout.readline()  # The runner prints its record (with the session) before launching.
        root = Path(json.loads(first)['session'])
        sessions[root] = child
        return root

    def launch(root):
        child = sessions.pop(root)
        for _ in child.stdout:
            pass
        return child.wait()

    return prepare, launch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--library', type=Path, help='The controller library (created from --seed when absent)')
    parser.add_argument('--seed', type=Path, default=skill_library.SEED)
    parser.add_argument('--tasks', required=True, help='Comma-separated tasks, in order')
    parser.add_argument('--records', type=Path, help='JSONL of finished tasks (default <library>-runs.jsonl)')
    parser.add_argument('--no-library', action='store_true', help='Run without a library (the comparison)')
    parser.add_argument('--min-success', type=float,
                        help='Publish only sessions whose formal share is at least this (default: always publish)')
    parser.add_argument('--runner', choices=('configure', 'deepseek'), default='configure')
    parser.add_argument('--configure-args', default='', help='configure runner: extra configure_auto_research.py flags')
    parser.add_argument('--launch-args', default='', help='configure runner: extra start_auto_research_agent.sh flags')
    parser.add_argument('--runner-args', default='', help='deepseek runner: extra run_deepseek_experiment.py flags')
    parser.add_argument('--dry-run', action='store_true', help='Print the plan and stop')
    args = parser.parse_args(argv)
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]
    if args.no_library:
        if not args.records:
            parser.error('--no-library needs --records')
    elif not args.library:
        parser.error('--library is required (or --no-library)')
    if args.min_success is not None and not 0 <= args.min_success <= 1:
        parser.error('--min-success must be between 0 and 1')
    args.library = args.library.resolve() if args.library else None
    args.records = (args.records or Path(f'{args.library}-runs.jsonl')).resolve()
    done = finished_tasks(read_records(args.records))
    print(json.dumps({'tasks': tasks, 'already_finished': sorted(done & set(tasks)),
                      'library': str(args.library) if not args.no_library else None,
                      'records': str(args.records), 'runner': args.runner, 'min_success': args.min_success},
                     indent=2), flush=True)
    if args.dry_run:
        return
    if not args.no_library:
        print('library commit', skill_library.init(args.library, args.seed), flush=True)
    prepare, launch = (deepseek_runner if args.runner == 'deepseek' else configure_runner)(args)
    rows = sequence(tasks, library=None if args.no_library else args.library, records=args.records,
                    prepare=prepare, launch=launch, publish=not args.no_library, min_success=args.min_success)
    raise SystemExit(1 if any(row['status'] == 'failed' for row in rows) else 0)


if __name__ == '__main__':
    main()
