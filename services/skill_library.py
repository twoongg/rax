"""A library of robot controllers that grows task by task.

    <library>/README.md               what the library is and where its seed came from
    <library>/INDEX.md                one row per entry, generated from the cards
    <library>/skills/<slug>/SKILL.md  the card (frontmatter + When to use, Procedure, Code,
                                      Task-specific constants, Failure modes)
    <library>/skills/<slug>/TASK.md   the task text the agent had
    <library>/skills/<slug>/demo/     the demonstration images, when the task had them
    <library>/skills/<slug>/code/     the controller as it ran: controller.py:main(ctx) and modules

An exploring agent gets a copy at `skill-library/` in its workspace (copy_into_workspace) and
the `skill-library` skill. After a session's formal batch, publish() adds the code that ran
there as a new entry, with its formal score on the card. The library on disk is a git
repository; each publish is a commit. Design: docs/superpowers/specs/2026-11-25-skill-library-design.md.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SEED = PROJECT/'skill_library_seed'/'robodojo'
WORKSPACE_DIR = 'skill-library'
PROVENANCE = 'skill-library.json'  # In the session root: which library and commit the agent got.
CARD = 'SKILL.md'
SLUG = re.compile(r'^[a-z0-9][a-z0-9-]{1,60}$')
# Files of a project larger than this stay out of the library (recordings, weights, frames).
MAX_FILE_BYTES = 5 * 1024**2
IGNORE = shutil.ignore_patterns('.git', '__pycache__', '*.pyc')
GIT = ('git', '-c', 'user.name=skill-library', '-c', 'user.email=skill-library@localhost',
       '-c', 'commit.gpgsign=false')
FRONTMATTER = re.compile(r'\A---\n(.*?)\n---\n', re.S)
FIELDS = ('name', 'description', 'status', 'robot', 'entry', 'task', 'source', 'formal', 'added')

INDEX_HEADER = """# Library index

One row per controller. Open the entries whose task is nearest to yours: compare their
`TASK.md` with yours, then read the card (`SKILL.md`) and the code it points to. `formal` is
the share of formal episodes the code completed (`not published` for the first four).

| Skill | Task | Formal | What it does | When to use |
|---|---|---|---|---|
"""


class NotPublishable(RuntimeError):
    """The session has no frozen code to publish."""


# --------------------------------------------------------------------------- cards

def frontmatter(text):
    """(fields, body) of a card; fields keep their order and are plain strings."""
    match = FRONTMATTER.match(text)
    if not match:
        return {}, text
    fields = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(':')
        if sep and key.strip():
            fields[key.strip()] = value.strip()
    return fields, text[match.end():]


def with_frontmatter(text, **changes):
    """The card with these fields set (others kept, known fields first in FIELDS order)."""
    fields, body = frontmatter(text)
    fields.update({key: str(value) for key, value in changes.items() if value is not None})
    keys = [k for k in FIELDS if k in fields] + [k for k in fields if k not in FIELDS]
    head = '\n'.join(f'{key}: {" ".join(fields[key].split())}' for key in keys)
    return f'---\n{head}\n---\n' + (body if body.startswith('\n') else '\n' + body)


def section(body, title):
    """The text under `## title`, up to the next `## ` heading."""
    match = re.search(rf'^## {re.escape(title)}\s*\n(.*?)(?=^## |\Z)', body, re.S | re.M)
    return match.group(1).strip() if match else ''


def first_sentence(text, limit=220):
    text = ' '.join(line for line in text.splitlines() if not line.lstrip().startswith(('|', '```')))
    text = ' '.join(text.split())
    end = re.search(r'(?<=[.!?])\s', text)
    sentence = text[:end.start()] if end else text
    return sentence if len(sentence) <= limit else sentence[:limit - 3].rstrip() + '...'


def entries(library):
    """Every entry with its card fields, in the order they were added (seed entries first)."""
    found = []
    for card in sorted((Path(library)/'skills').glob(f'*/{CARD}')):
        fields, body = frontmatter(card.read_text())
        found.append({'slug': card.parent.name, 'fields': fields,
                      'when': first_sentence(section(body, 'When to use'))})
    return sorted(found, key=lambda e: (e['fields'].get('added', ''), e['slug']))


def _cell(text):
    return ' '.join(str(text).split()).replace('|', '\\|')


def write_index(library):
    rows = [f'| [{e["slug"]}](skills/{e["slug"]}/{CARD}) | `{_cell(e["fields"].get("task", ""))}` '
            f'| {_cell(e["fields"].get("formal", "unknown"))} | {_cell(e["fields"].get("description", ""))} '
            f'| {_cell(e["when"])} |' for e in entries(library)]
    (Path(library)/'INDEX.md').write_text(INDEX_HEADER + ''.join(row + '\n' for row in rows))


# --------------------------------------------------------------------------- git

def _git(library, *args, check=True):
    return subprocess.run([*GIT, '-C', str(library), *args], check=check, capture_output=True, text=True)


def head(library):
    """The library's current commit, or None when it is not a git repository."""
    if not (Path(library)/'.git').exists():
        return None
    result = _git(library, 'rev-parse', 'HEAD', check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def commit(library, message):
    """Commit every change; returns the new commit, or the old one when nothing changed."""
    if not (Path(library)/'.git').exists():
        return None
    _git(library, 'add', '-A')
    if _git(library, 'diff', '--cached', '--quiet', check=False).returncode != 0:
        _git(library, 'commit', '-q', '-m', message)
    return head(library)


def init(library, seed=SEED):
    """Create the library from the seed as a git repository; an existing library is kept."""
    library = Path(library)
    if (library/'INDEX.md').is_file():
        return head(library)
    if library.exists() and any(library.iterdir()):
        raise ValueError(f'{library} exists and is not a library (no INDEX.md)')
    shutil.copytree(seed, library, dirs_exist_ok=True, ignore=IGNORE)
    write_index(library)
    _git(library, 'init', '-q')
    return commit(library, f'Seed the library from {Path(seed).name}')


# --------------------------------------------------------------------------- workspace

def copy_into_workspace(library, workspace):
    """Copy the library (without .git) to <workspace>/skill-library; returns its provenance."""
    library = Path(library)
    if not (library/'INDEX.md').is_file():
        raise ValueError(f'{library} is not a skill library (no INDEX.md)')
    target = Path(workspace)/WORKSPACE_DIR
    shutil.copytree(library, target, ignore=IGNORE)
    return {'library': str(library), 'commit': head(library),
            'entries': [e['slug'] for e in entries(library)], 'copied_at': time.time()}


# --------------------------------------------------------------------------- publishing

def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def formal_outcome(root):
    """(successes, episodes) of the session's formal batch (code mode) or agent formal, or None."""
    summary = _json(Path(root)/'formal.json', {}) or {}
    for key in ('bundle_formal', 'agent_formal'):
        outcome = summary.get(key)
        if not isinstance(outcome, dict):
            continue
        if outcome.get('episode_count'):
            return int(outcome.get('success_count') or 0), int(outcome['episode_count'])
        if isinstance(outcome.get('task_complete'), bool):
            return int(outcome['task_complete']), 1
    return None


def _package(root, state):
    checkpoint = state.get('exploration_checkpoint')
    if not checkpoint:
        return None
    return Path(root)/'bundles'/state['bundles'][checkpoint['bundle']]['directory']


def _project(code):
    """The controller project inside a frozen code/ folder."""
    projects = sorted(p for p in code.iterdir() if p.is_dir() and not p.name.startswith(('.', '__')))
    if not projects:
        return code
    for test in (lambda p: (p/CARD).is_file() and (p/'controller.py').is_file(),
                 lambda p: (p/'controller.py').is_file()):
        chosen = [p for p in projects if test(p)]
        if len(chosen) == 1:
            return chosen[0]
    return projects[0] if len(projects) == 1 else code


def frozen_code(root):
    """(code directory, card candidates) of what the session's evaluation ran.

    Code mode: the submitted bundle, exactly what the formal batch ran; its card falls back to
    the same project in the exploration freeze. Live copies: the project in the freeze's code/.
    """
    root = Path(root)
    state = _json(root/'state.json')
    if not state:
        raise NotPublishable('no state.json')
    package = _package(root, state)
    submission = state.get('submission')
    if submission:
        manifest = state['bundles'][submission['bundle']]
        code = root/'bundles'/manifest['directory']
        cards = [code/CARD]
        if package is not None:
            cards.append(package/'transfer'/manifest.get('workspace_path', 'code/controller')/CARD)
        return code, cards
    if package is None or not (package/'transfer'/'code').is_dir():
        raise NotPublishable('exploration did not close with frozen code')
    project = _project(package/'transfer'/'code')
    return project, [project/CARD]


def _definitions(path):
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return [], []
    parts, constants = [], []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
            doc = (ast.get_docstring(node) or '').strip().splitlines()
            parts.append((node.name, doc[0] if doc else ''))
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            constants.append(ast.unparse(node)[:100])
    return parts, constants


def stub_card(code, slug, task, robot):
    """A card made from the code when the agent wrote none."""
    rows, constants = [], []
    for path in sorted(Path(code).rglob('*.py')):
        relative = path.relative_to(code).as_posix()
        parts, values = _definitions(path)
        rows += [f'| `{name}` | {relative} | {_cell(doc)} |' for name, doc in parts]
        constants += [f'- {relative}: `{value}`' for value in values]
    table = '\n'.join(['| Part | Where | First docstring line |', '|---|---|---|', *rows]) if rows else 'No Python functions found.'
    return (f'---\nname: {slug}\ndescription: Controller for {task}; no card was written, so this one lists the code.\n'
            f'status: draft\nrobot: {robot}\nentry: code/controller.py\ntask: {task}\n---\n\n# {slug}\n\n'
            f'## When to use\n\nFor a task like the one in [TASK.md](TASK.md). No card was written: read '
            f'`code/controller.py` to judge what carries over.\n\n## Procedure\n\nSee `main(ctx)` in '
            f'`code/controller.py`.\n\n## Code\n\n{table}\n\n## Task-specific constants\n\n'
            + ('\n'.join(constants) if constants else 'None found at module level.')
            + '\n\n## Failure modes\n\nNot recorded.\n')


def _slug(library, task):
    base = re.sub(r'[^a-z0-9-]+', '-', task.lower().replace('_', '-')).strip('-')[:56] or 'controller'
    slug, number = base, 2
    while (Path(library)/'skills'/slug).exists():
        slug, number = f'{base}-{number}', number + 1
    return slug


def _copy_code(code, target):
    """Copy the project's files, without its card and without files over MAX_FILE_BYTES."""
    left_out = []
    for path in sorted(Path(code).rglob('*')):
        relative = path.relative_to(code)
        if (path.is_symlink() or not path.is_file() or relative.as_posix() == CARD
                or '__pycache__' in relative.parts or path.suffix == '.pyc'):
            continue
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            left_out.append(f'{relative.as_posix()} ({size / 1024**2:.1f} MB)')
            continue
        (target/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target/relative)
    return left_out


def publish(root, library, *, min_success=None, now=None):
    """Add the session's evaluated code to the library as a new entry.

    Returns {'status': 'published', 'slug', 'formal', 'commit'} or {'status': 'skipped', 'reason'}.
    min_success: publish only when the formal share is at least this (None: always).
    """
    root, library = Path(root), Path(library)
    if not (library/'INDEX.md').is_file():
        raise ValueError(f'{library} is not a skill library (no INDEX.md)')
    config = _json(root/'operator-input.json', {}) or {}
    task = config.get('task') or root.name
    outcome = formal_outcome(root)
    formal = f'{outcome[0]}/{outcome[1]}' if outcome else 'unknown'
    if min_success is not None and (outcome is None or outcome[0] / outcome[1] < min_success):
        return {'status': 'skipped', 'reason': f'formal {formal} is below {min_success}', 'formal': formal}
    try:
        code, cards = frozen_code(root)
    except NotPublishable as error:
        return {'status': 'skipped', 'reason': str(error), 'formal': formal}
    slug = _slug(library, task)
    entry = library/'skills'/slug
    (entry/'code').mkdir(parents=True)
    left_out = _copy_code(code, entry/'code')
    robot = config.get('benchmark', 'robodojo')
    card_path = next((path for path in cards if path.is_file()), None)
    card = card_path.read_text() if card_path else stub_card(entry/'code', slug, task, robot)
    if not frontmatter(card)[0]:
        card = f'---\nname: {slug}\n---\n\n' + card
    fields = frontmatter(card)[0]
    entry_file = 'code/controller.py' if (entry/'code'/'controller.py').is_file() else fields.get('entry')
    agent = ', '.join(filter(None, (config.get('agent_cli'), config.get('agent_provider'))))
    card = with_frontmatter(
        card, name=slug, description=fields.get('description') or f'Controller for {task}.',
        status=fields.get('status') or 'draft', robot=fields.get('robot') or robot, entry=entry_file, task=task,
        source=f'session {root.name}' + (f' ({agent})' if agent else ''),
        formal=formal, added=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)))
    if left_out:
        card = card.rstrip('\n') + '\n\n## Left out of this entry\n\n' + ''.join(f'- {name}\n' for name in left_out)
    (entry/CARD).write_text(card)
    workspace = root/'agent-workspace'
    if (workspace/'TASK.md').is_file():
        (entry/'TASK.md').write_text((workspace/'TASK.md').read_text().replace('runtime/demonstrations/', 'demo/'))
    demonstrations = workspace/'runtime'/'demonstrations'
    if demonstrations.is_dir() and any(demonstrations.iterdir()):
        shutil.copytree(demonstrations, entry/'demo', ignore=IGNORE)
    write_index(library)
    revision = commit(library, f'Add {slug} from {root.name} (formal {formal})')
    return {'status': 'published', 'slug': slug, 'formal': formal, 'commit': revision,
            **({'left_out': left_out} if left_out else {})}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Manage a controller library.')
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('init', help='create a library from the seed (a git repository)')
    make.add_argument('library', type=Path)
    make.add_argument('--seed', type=Path, default=SEED)
    index = commands.add_parser('index', help='regenerate INDEX.md from the cards')
    index.add_argument('library', type=Path)
    add = commands.add_parser('publish', help="add a finished session's evaluated code")
    add.add_argument('session', type=Path)
    add.add_argument('library', type=Path)
    add.add_argument('--min-success', type=float)
    args = parser.parse_args(argv)
    if args.command == 'init':
        print(init(args.library, args.seed))
    elif args.command == 'index':
        write_index(args.library)
    else:
        print(json.dumps(publish(args.session, args.library, min_success=args.min_success), indent=2))


if __name__ == '__main__':
    main()
