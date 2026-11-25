"""The controller library: seed, workspace copy, publishing after the formal batch, and the
task sequence (services/skill_library.py, scripts/run_skill_library_sequence.py)."""
import json
import py_compile
import shutil
import sys
from pathlib import Path

import pytest

from services import skill_library as lib

PROJECT = Path(__file__).resolve().parents[2]
SECTIONS = ('When to use', 'Procedure', 'Code', 'Task-specific constants', 'Failure modes')


def test_seed_cards_name_their_entry_task_and_sections():
    slugs = {path.name for path in (lib.SEED/'skills').iterdir()}
    assert slugs == {'make-kong', 'plug-in-charger', 'pour-by-language', 'swap-blocks'}
    for slug in slugs:
        entry = lib.SEED/'skills'/slug
        fields, body = lib.frontmatter((entry/'SKILL.md').read_text())
        assert fields['name'] == slug and lib.SLUG.match(slug)
        assert fields['task'] == slug.replace('-', '_') and fields['formal'] == 'not published'
        assert (entry/fields['entry']).is_file() and fields['entry'] == 'code/controller.py'
        assert all(lib.section(body, title) for title in SECTIONS), slug
        assert f"Task: `{fields['task']}`" in (entry/'TASK.md').read_text()


def test_seed_code_compiles_and_the_index_is_current(tmp_path):
    for path in lib.SEED.rglob('*.py'):
        py_compile.compile(str(path), cfile=str(tmp_path/'x.pyc'), doraise=True)
    copy = tmp_path/'seed'
    shutil.copytree(lib.SEED, copy)
    lib.write_index(copy)
    assert (copy/'INDEX.md').read_text() == (lib.SEED/'INDEX.md').read_text()


def test_init_makes_a_git_library_once(tmp_path):
    library = tmp_path/'library'
    first = lib.init(library)
    assert first and (library/'.git').is_dir() and (library/'INDEX.md').is_file()
    assert lib.init(library) == first  # An existing library is kept.
    (tmp_path/'other').mkdir()
    (tmp_path/'other'/'file').write_text('x')
    with pytest.raises(ValueError, match='not a library'):
        lib.init(tmp_path/'other')


def make_session(root, *, task='pour_liquid_into_cup', mode='code', formal=(3, 10), card=None, freeze_card=None,
                 files=None, demonstrations=True):
    """A finished session's files as the launcher leaves them (only what publish reads)."""
    root.mkdir(parents=True)
    (root/'operator-input.json').write_text(json.dumps({'task': task, 'agent_cli': 'claude',
                                                        'agent_provider': 'custom', 'benchmark': 'robodojo'}))
    files = files or {'controller.py': 'def main(ctx):\n    """Do the task."""\n\n\nGRASP_Z = 0.08\n',
                      'helper.py': 'def find(obs):\n    """Find the thing."""\n'}
    state = {'bundles': {'pkg': {'directory': 'pkg'}}, 'exploration_checkpoint': {'bundle': 'pkg'}}
    frozen = root/'bundles'/'pkg'/'transfer'/'code'/'proj'
    frozen.mkdir(parents=True)
    for name, text in files.items():
        (frozen/name).write_text(text)
    if freeze_card:
        (frozen/'SKILL.md').write_text(freeze_card)
    if mode == 'code':
        bundle = root/'bundles'/'b1'
        bundle.mkdir()
        for name, text in files.items():
            (bundle/name).write_text(text)
        if card:
            (bundle/'SKILL.md').write_text(card)
        state['bundles']['b1'] = {'directory': 'b1', 'workspace_path': 'code/proj'}
        state['submission'] = {'bundle': 'b1'}
    elif mode == 'open':
        state = {'bundles': {}}
    (root/'state.json').write_text(json.dumps(state))
    if formal is not None:
        key = 'bundle_formal' if mode == 'code' else 'agent_formal'
        (root/'formal.json').write_text(json.dumps({'status': 'completed', key: {
            'status': 'completed', 'success_count': formal[0], 'episode_count': formal[1]}}))
    workspace = root/'agent-workspace'
    (workspace/'runtime'/'demonstrations').mkdir(parents=True)
    (workspace/'TASK.md').write_text(f'Task: `{task}`\nRead runtime/demonstrations/manifest.json\n')
    if demonstrations:
        (workspace/'runtime'/'demonstrations'/'manifest.json').write_text('{}')
        (workspace/'runtime'/'demonstrations'/'terminal.jpg').write_bytes(b'jpg')
    return root


CARD = """---
name: whatever
description: Pick the thing and put it down.
---

# whatever

## When to use

A task that picks a thing. More text.

## Procedure

1. Go.
"""


def test_publish_adds_the_submitted_bundle_with_its_score(tmp_path, monkeypatch):
    library = tmp_path/'library'
    before = lib.init(library)
    monkeypatch.setattr(lib, 'MAX_FILE_BYTES', 100)
    files = {'controller.py': 'def main(ctx):\n    pass\n', 'weights.bin': 'x' * 200}
    root = make_session(tmp_path/'s1', card=CARD, files=files)
    result = lib.publish(root, library, now=0)
    assert result['status'] == 'published' and result['slug'] == 'pour-liquid-into-cup'
    assert result['formal'] == '3/10' and result['commit'] not in (None, before)
    assert result['left_out'] == ['weights.bin (0.0 MB)']
    entry = library/'skills'/'pour-liquid-into-cup'
    assert sorted(p.relative_to(entry).as_posix() for p in entry.rglob('*') if p.is_file()) == [
        'SKILL.md', 'TASK.md', 'code/controller.py', 'demo/manifest.json', 'demo/terminal.jpg']
    fields, body = lib.frontmatter((entry/'SKILL.md').read_text())
    assert fields['name'] == 'pour-liquid-into-cup' and fields['task'] == 'pour_liquid_into_cup'
    assert fields['formal'] == '3/10' and fields['source'] == 'session s1 (claude, custom)'
    assert fields['description'] == 'Pick the thing and put it down.' and fields['status'] == 'draft'
    assert fields['entry'] == 'code/controller.py' and fields['added'] == '1970-01-01T00:00:00Z'
    assert '- weights.bin (0.0 MB)' in body
    assert 'demo/manifest.json' in (entry/'TASK.md').read_text()
    row = next(line for line in (library/'INDEX.md').read_text().splitlines() if 'pour-liquid-into-cup' in line)
    assert '| 3/10 |' in row and 'A task that picks a thing.' in row
    # The seed rows come first; the new entry is last.
    assert (library/'INDEX.md').read_text().rstrip().endswith(row)


def test_a_card_left_only_in_the_freeze_is_used(tmp_path):
    library = tmp_path/'library'
    lib.init(library)
    root = make_session(tmp_path/'s1', freeze_card=CARD)
    lib.publish(root, library)
    body = (library/'skills'/'pour-liquid-into-cup'/'SKILL.md').read_text()
    assert 'A task that picks a thing.' in body


def test_live_copies_publish_the_frozen_project_with_a_stub_card(tmp_path):
    library = tmp_path/'library'
    lib.init(library)
    root = make_session(tmp_path/'s1', mode='live', formal=(0, 5), demonstrations=False)
    result = lib.publish(root, library)
    assert result['status'] == 'published' and result['formal'] == '0/5'
    entry = library/'skills'/'pour-liquid-into-cup'
    card = (entry/'SKILL.md').read_text()
    assert '| `main` | controller.py | Do the task. |' in card and '| `find` | helper.py | Find the thing. |' in card
    assert '`GRASP_Z = 0.08`' in card and 'no card was written' in card
    assert not (entry/'demo').exists() and (entry/'code'/'helper.py').is_file()


def test_a_repeated_task_gets_its_own_entry(tmp_path):
    library = tmp_path/'library'
    lib.init(library)
    assert lib.publish(make_session(tmp_path/'s1'), library)['slug'] == 'pour-liquid-into-cup'
    assert lib.publish(make_session(tmp_path/'s2'), library)['slug'] == 'pour-liquid-into-cup-2'
    assert [e['slug'] for e in lib.entries(library)][-2:] == ['pour-liquid-into-cup', 'pour-liquid-into-cup-2']


def test_min_success_and_missing_freeze_publish_nothing(tmp_path):
    library = tmp_path/'library'
    before = lib.init(library)
    low = lib.publish(make_session(tmp_path/'s1', formal=(1, 10)), library, min_success=0.3)
    assert low == {'status': 'skipped', 'reason': 'formal 1/10 is below 0.3', 'formal': '1/10'}
    unknown = lib.publish(make_session(tmp_path/'s2', formal=None), library, min_success=0.3)
    assert unknown['status'] == 'skipped'
    open_ = lib.publish(make_session(tmp_path/'s3', mode='open', formal=None), library)
    assert open_['status'] == 'skipped' and 'did not close' in open_['reason']
    assert lib.head(library) == before and not any((library/'skills').glob('pour-liquid*'))
    # Without a threshold, a zero score is published like any other.
    assert lib.publish(make_session(tmp_path/'s4', formal=(0, 10)), library)['status'] == 'published'


def populate(tmp_path, library, *, formal_episode=False):
    sys.path.insert(0, str(PROJECT/'scripts'))
    import render_workspaces
    from harness.codex_cli.auto_research import populate_workspace
    from services.mcp_contract import Contract
    config = render_workspaces.configuration(tmp_path/'root', skill_library=str(library) if library else None,
                                             library_sequence='seq' if library else None,
                                             library_step='1/2' if library else None, agent_formal=formal_episode)
    workspace = tmp_path/'ws'
    workspace.mkdir()
    contract = Contract.from_config(config, isolated=True) if formal_episode else None
    populate_workspace(config, workspace, contract=contract, formal_episode=formal_episode)
    return config, workspace


def test_the_exploring_agent_gets_a_copy_of_the_library(tmp_path):
    library = tmp_path/'library'
    commit = lib.init(library)
    config, workspace = populate(tmp_path, library)
    assert (workspace/'skill-library'/'INDEX.md').is_file() and not (workspace/'skill-library'/'.git').exists()
    assert (workspace/'.agents/skills/skill-library/SKILL.md').is_file()
    agents = (workspace/'AGENTS.md').read_text()
    assert '0. **Start from the controller library.**' in agents and 'skill-library/INDEX.md' in agents
    # Code first, in a fresh session: no whole-task manual solve before automating.
    words = ' '.join(agents.split())
    assert 'This session started fresh for this task' in words and 'Solve the task by hand first' not in words
    assert 'If manual success is not yet' not in words and 'Work out by hand what the copied code does not do' in words
    provenance = json.loads((config.root/lib.PROVENANCE).read_text())
    assert provenance['commit'] == commit and 'swap-blocks' in provenance['entries']
    assert (provenance['sequence'], provenance['step']) == ('seq', '1/2')


@pytest.mark.parametrize('case', ['no library', 'formal copy'])
def test_no_library_without_the_option_or_in_a_formal_copy(tmp_path, case):
    library = tmp_path/'library'
    lib.init(library)
    _, workspace = populate(tmp_path, None if case == 'no library' else library,
                            formal_episode=case == 'formal copy')
    assert not (workspace/'skill-library').exists()
    assert not (workspace/'.agents/skills/skill-library').exists()
    assert 'skill-library' not in (workspace/'AGENTS.md').read_text()
    assert 'started fresh' not in (workspace/'AGENTS.md').read_text()


def test_the_sequence_publishes_in_order_stops_on_a_broken_session_and_resumes(tmp_path):
    sys.path.insert(0, str(PROJECT/'scripts'))
    import run_skill_library_sequence as runner
    library, records = tmp_path/'library', tmp_path/'runs.jsonl'
    lib.init(library)
    broken = {'b'}
    seen = []

    def prepare(task, step):
        root = tmp_path/'sessions'/f'{task}-{len(seen)}'
        seen.append((task, step))
        make_session(root, task=task, formal=None)
        return root

    def launch(root):
        task = json.loads((root/'operator-input.json').read_text())['task']
        if task not in broken:
            (root/'formal.json').write_text(json.dumps({'bundle_formal': {'success_count': 1, 'episode_count': 2}}))
        return 0

    rows = runner.sequence(['a', 'b', 'c'], library=library, records=records, prepare=prepare, launch=launch,
                           log=lambda *a: None)
    assert [(r['task'], r['status']) for r in rows] == [('a', 'published'), ('b', 'failed')]
    assert rows[0]['formal'] == '1/2' and rows[0]['library_after'] != rows[0]['library_before']
    broken.clear()
    rows = runner.sequence(['a', 'b', 'c'], library=library, records=records, prepare=prepare, launch=launch,
                           log=lambda *a: None)
    assert [(r['task'], r['status']) for r in rows] == [('b', 'published'), ('c', 'published')]
    # Every attempt is a new session; a rerun never resumes one, and steps keep their place.
    assert seen == [('a', '1/3'), ('b', '2/3'), ('b', '2/3'), ('c', '3/3')]
    assert len({r['session'] for r in runner.read_records(records)}) == 4
    assert [e['slug'] for e in lib.entries(library)][-3:] == ['a', 'b', 'c']
    assert len(runner.read_records(records)) == 4


def test_the_comparison_run_records_scores_without_a_library(tmp_path):
    sys.path.insert(0, str(PROJECT/'scripts'))
    import run_skill_library_sequence as runner

    def prepare(task, step):
        return make_session(tmp_path/task, task=task, formal=(2, 4))

    rows = runner.sequence(['a'], library=None, records=tmp_path/'runs.jsonl', prepare=prepare,
                           launch=lambda root: 0, publish=False, log=lambda *a: None)
    assert rows[0]['status'] == 'evaluated' and rows[0]['formal'] == '2/4' and 'publish' not in rows[0]


def test_the_dashboard_shows_what_a_library_session_was_given_and_what_piled_up(tmp_path):
    from services.dashboard.sessions import ResearchSessions
    library = tmp_path/'library'
    commit = lib.init(library)
    dashboard = ResearchSessions(tmp_path/'project', lambda: ['make_kong'])
    ident = '20261125T120000Z_0123456789'
    root = make_session(dashboard.root/ident, formal=(2, 5))
    config = json.loads((root/'operator-input.json').read_text())
    (root/'operator-input.json').write_text(json.dumps({**config, 'exploration_seeds': [1], 'sim_gpu': 'GPU-a',
                                                        'skill_library': str(library), 'library_sequence': 'seq',
                                                        'library_step': '2/6'}))
    (root/lib.PROVENANCE).write_text(json.dumps({'library': str(library), 'commit': commit,
                                                 'entries': [e['slug'] for e in lib.entries(library)]}))
    lib.publish(root, library, now=100)
    lib.publish(make_session(tmp_path/'later', task='swap_t', formal=(1, 5)), library, now=200)
    detail = dashboard.detail(ident)
    assert detail['research'] == {'mode': 'library sequence', 'sequence': 'seq', 'step': '2/6'}
    view = detail['skill_library']
    assert view['available'] and view['given_commit'] == commit and view['given_count'] == 4
    assert view['head'] == lib.head(library) != commit
    assert [(e['slug'], e['given'], e['from_this_session'], e['formal']) for e in view['entries']] == [
        ('make-kong', True, False, 'not published'), ('plug-in-charger', True, False, 'not published'),
        ('pour-by-language', True, False, 'not published'), ('swap-blocks', True, False, 'not published'),
        ('pour-liquid-into-cup', False, True, '2/5'), ('swap-t', False, False, '1/5')]
    # A session without a library has no library view; an unreadable library says so.
    other = make_session(dashboard.root/'20261125T120001Z_0123456789')
    config = json.loads((other/'operator-input.json').read_text())
    (other/'operator-input.json').write_text(json.dumps({**config, 'exploration_seeds': [1], 'sim_gpu': 'GPU-a'}))
    assert dashboard.detail(other.name)['skill_library'] is None
    assert dashboard.detail(other.name)['research'] == {'mode': 'single task'}
    (other/'operator-input.json').write_text(json.dumps({**config, 'exploration_seeds': [1], 'sim_gpu': 'GPU-a',
                                                         'skill_library': str(tmp_path/'gone')}))
    gone = dashboard.detail(other.name)['skill_library']
    assert gone['available'] is False and gone['entries'] == [] and gone['given_count'] is None


def test_a_server_only_module_is_flagged_until_added_to_its_entry(tmp_path):
    # The repository's seed lacks robodojo_toolkit.py; a server's seed may hold it.
    seed = tmp_path/'seed'
    shutil.copytree(lib.SEED, seed)
    (seed/'skills/make-kong/code/robodojo_toolkit.py').unlink(missing_ok=True)
    library = tmp_path/'library'
    seeded = lib.init(library, seed)
    assert lib.missing_modules(library) == {'make-kong': ['robodojo_toolkit']}
    assert 'add-file' in lib.missing_report(library)[0]
    toolkit = tmp_path/'server'/'robodojo_toolkit.py'
    toolkit.parent.mkdir()
    toolkit.write_text('import numpy\n\n\ndef parse_reply(reply):\n    return {}, {}\n')
    added = lib.add_files(library, 'make-kong', [toolkit])
    assert added != seeded and (library/'skills/make-kong/code/robodojo_toolkit.py').read_text() == toolkit.read_text()
    assert lib.missing_modules(library) == {} and lib.missing_report(library) == []
    assert lib.add_files(library, 'make-kong', [toolkit]) == added  # The same file again: no new commit.
    with pytest.raises(ValueError, match='no entry'):
        lib.add_files(library, 'no-such-entry', [toolkit])
    # The sequence script takes the same as --add-file SLUG=PATH.
    sys.path.insert(0, str(PROJECT/'scripts'))
    import run_skill_library_sequence as runner
    assert runner.add_files(library, [f'make-kong={toolkit}']) == added
    with pytest.raises(ValueError, match='SLUG=PATH'):
        runner.add_files(library, [str(toolkit)])


def test_only_the_sequence_script_makes_library_sessions(tmp_path, monkeypatch, capsys):
    # The configure flag alone is refused before anything else runs.
    sys.path.insert(0, str(PROJECT/'scripts'))
    import configure_auto_research as configure
    monkeypatch.setattr(sys, 'argv', ['configure_auto_research.py', '--task', 'make_kong', '--sim-gpu', '0',
                                      '--skill-library', str(lib.SEED)])
    with pytest.raises(SystemExit):
        configure.main()
    assert 'for single-task research, leave it out' in capsys.readouterr().err
    # The configuration itself holds the three together, as K/N, and never with an adaptation source.
    import render_workspaces
    for changes, match in (({'skill_library': str(lib.SEED)}, 'go together'),
                           ({'library_sequence': 'seq', 'library_step': '1/2'}, 'go together'),
                           ({'skill_library': str(lib.SEED), 'library_sequence': 'seq', 'library_step': '3/2'}, 'K/N'),
                           ({'skill_library': str(lib.SEED), 'library_sequence': 'seq', 'library_step': '1/2',
                             'adaptation_source': str(tmp_path)}, 'fresh agent')):
        with pytest.raises(ValueError, match=match):
            render_workspaces.configuration(tmp_path/'root', **changes)
    config = render_workspaces.configuration(tmp_path/'root', skill_library=str(lib.SEED), library_sequence='seq',
                                             library_step='2/2')
    assert config.library_step == '2/2'


def test_a_library_session_saved_before_sequences_still_loads(tmp_path):
    import dataclasses
    sys.path.insert(0, str(PROJECT/'scripts'))
    import render_workspaces
    from services.controller.config import Configuration
    saved = json.loads(json.dumps(dataclasses.asdict(render_workspaces.configuration(tmp_path/'root')), default=str))
    saved.pop('library_sequence'), saved.pop('library_step')
    saved['skill_library'] = str(lib.SEED)
    config = Configuration.from_dict(saved)
    assert (config.library_sequence, config.library_step) == ('unnamed', '1/1')
