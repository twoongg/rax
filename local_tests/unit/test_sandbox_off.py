"""Sessions with the sandbox off, for hosts where bubblewrap cannot create namespaces (such as
Kubernetes pods): Claude Code runs the agent's commands without its sandbox, and bundles run as
plain processes with the workspace, exports, environment and limits they rely on."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_controller_backend import configuration
from harness.claude_cli import auto_research as claude
from services.controller.config import Configuration, Limits
from services.controller.gateway import Gateway
from services.controller.sandbox import BundleSandbox, UnconfinedBundle, bundle_runner
from services.controller.storage import snapshot
from services.controller.supervisor import Supervisor


def unsandboxed(tmp_path):
    return replace(configuration(tmp_path), agent_cli='claude', sandbox=False)


def test_the_sandbox_is_on_unless_a_claude_session_turns_it_off(tmp_path):
    config = configuration(tmp_path)
    assert config.sandbox and bundle_runner(config) is BundleSandbox
    assert bundle_runner(unsandboxed(tmp_path)) is UnconfinedBundle
    with pytest.raises(ValueError, match='only available with Claude Code'):
        replace(config, sandbox=False)  # Codex brings its own sandbox.
    with pytest.raises(ValueError, match='sandbox must be true or false'):
        replace(unsandboxed(tmp_path), sandbox='off')
    # A session saved before the setting existed resumes in the sandbox.
    saved = json.loads(json.dumps(asdict(config), default=str))
    del saved['sandbox']
    assert Configuration.from_dict(saved).sandbox


def test_the_supervisor_runs_bundles_unconfined_only_with_the_sandbox_off(tmp_path):
    for config, runner in ((configuration(tmp_path/'on'), BundleSandbox),
                           (unsandboxed(tmp_path/'off'), UnconfinedBundle)):
        supervisor = Supervisor(config, lambda: None)
        try:
            assert supervisor.runner_factory is runner
        finally:
            supervisor.close()


def test_claude_without_its_sandbox_runs_commands_unprompted(tmp_path):
    config = unsandboxed(tmp_path)
    workspace = tmp_path/'agent-workspace'
    workspace.mkdir()
    deployed = {'workspace': workspace}
    settings = claude.settings(config, deployed, tmp_path/'socket/mcp.sock')
    assert settings['sandbox'] == {'enabled': False}
    permissions = settings['permissions']
    assert {'Bash', 'mcp__auto_research'} <= set(permissions['allow'])
    assert {'WebFetch', 'WebSearch'} <= set(permissions['deny']) and permissions['defaultMode'] == 'acceptEdits'
    assert any(rule.startswith('Edit(') for rule in permissions['deny'])  # Claude's own settings stay guarded.
    sandboxed = claude.settings(replace(config, sandbox=True), deployed, tmp_path/'socket/mcp.sock')
    assert sandboxed['sandbox']['enabled'] and sandboxed['sandbox']['failIfUnavailable']
    assert 'Bash' not in sandboxed['permissions']['allow']


def gateway_for(backend):
    gateway = Gateway(backend, None, audit=lambda _: None, charge=lambda *_: None,
                      refund=lambda *_: None, budget=lambda: {})
    return gateway, gateway.acquire()


class Backend:
    step = 0

    def tools(self):
        return []

    def call(self, name, arguments):
        if name == 'robodojo_step':
            self.step += 1
        return {'step_id': self.step}, []


@pytest.fixture(autouse=True)
def host_limits(monkeypatch):
    """The resource limits are Linux's; macOS refuses them, as it does for the sandbox."""
    if sys.platform != 'linux':
        monkeypatch.setattr(BundleSandbox, 'limit', lambda self: None)


def bundle(tmp_path, source):
    src = tmp_path/'source'
    src.mkdir()
    (src/'controller.py').write_text(source)
    (src/'data.txt').write_text('bundled')
    snapshot(src, tmp_path/'bundle', max_bytes=100000)
    return tmp_path/'bundle'


def test_a_bundle_runs_in_its_workspace_and_exports_its_output(tmp_path, monkeypatch):
    monkeypatch.setenv('OPERATOR_SECRET', 'x')
    controller = bundle(tmp_path, '''
import json, os, subprocess, sys, tempfile
from pathlib import Path
def main(ctx):
    assert Path("code/controller/controller.py").is_file() and Path("api/runtime.py").is_file()
    assert Path(__file__).with_name("data.txt").read_text() == "bundled"
    assert os.environ["HOME"] == os.getcwd() and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert "OPERATOR_SECRET" not in os.environ and set(os.environ) >= {"PYTHONPATH", "OMP_NUM_THREADS"}
    assert Path(tempfile.gettempdir()).name == "tmp" and Path(tempfile.gettempdir()).parent != Path("/")
    for i in range(2):
        state = json.loads(ctx.call("robodojo_observe")["content"][0]["text"])
        assert state["step_id"] == i
        ctx.call("robodojo_step", actions=[[0.0]*14])
    print("python-print-checkpoint")
    subprocess.run([sys.executable, "-c", "print('child-print-checkpoint')"], check=True)
    Path("output/report.json").write_text(json.dumps({"unconfined": True}))
''')
    backend = Backend()
    gateway, token = gateway_for(backend)
    runner = UnconfinedBundle(Limits(30, 2048, 1, 32, 16, 100000))
    runner.preflight()
    result = runner.run(controller, mode='rehearsal', entrypoint='controller.py', gateway=gateway, token=token,
                        output=tmp_path/'run', workspace=tmp_path/'agent-workspace')
    run = tmp_path/'run'
    assert result['returncode'] == 0, (run/'stderr.log').read_text()
    assert backend.step == 2
    assert json.loads((run/'files/report.json').read_text()) == {'unconfined': True}
    assert 'python-print-checkpoint' in (run/'stdout.log').read_text()
    assert 'child-print-checkpoint' in (run/'stderr.log').read_text()
    # The frozen bundle is untouched; the run's links, scratch and writable copy are gone.
    assert sorted(p.name for p in controller.iterdir()) == ['controller.py', 'data.txt']
    assert not any((run/name).exists() for name in ('workspace', 'tmp', 'writable'))
    assert not (tmp_path/'agent-workspace').exists()


def alive(pid):
    state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith('Z')


@pytest.mark.parametrize('ending, reason', [('time.sleep(300)', 'timeout'), ('return', 'exit')])
def test_a_bundle_leaves_no_process_behind(tmp_path, ending, reason):
    # Without a PID namespace, a background child would outlive a controller that returned.
    controller = bundle(tmp_path, f'''
import subprocess, time
from pathlib import Path
def main(ctx):
    child = subprocess.Popen(["sleep", "300"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    Path("output/child.pid").write_text(str(child.pid))
    {ending}
''')
    gateway, token = gateway_for(Backend())
    runner = UnconfinedBundle(Limits(3, 2048, 1, 32, 16, 100000))
    result = runner.run(controller, mode='rehearsal', entrypoint='controller.py', gateway=gateway, token=token,
                        output=tmp_path/'run', workspace=tmp_path/'agent-workspace')
    assert result['reason'] == reason, (tmp_path/'run/stderr.log').read_text()
    child = int((tmp_path/'run/files/child.pid').read_text())
    deadline = time.monotonic() + 5
    while alive(child) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(child)
    assert not (tmp_path/'run/writable').exists()
