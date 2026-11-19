"""gemini_generate served by a session's own OpenAI-compatible server (perception_url)."""
from dataclasses import replace
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from services.controller.config import Configuration, retire_fields
from services.controller.gateway import GEMINI_DESCRIPTION, Gateway, gemini_description
from services.controller.gemini import MODEL, GeminiRouter, chat_endpoint, router_for
from services.workspace import Setup, build, recipe_for
from services.mcp_contract import Contract
from test_controller_backend import Backend, Runner, configuration, packet, value
from test_gemini_images import image_block, picture

SERVED = 'open-vlm'
QUESTION = [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is in the image?'},
                                         image_block(picture())]}]


class Server:
    """A local chat-completions endpoint that records what it was sent."""

    def __init__(self, model=SERVED):
        requests = self.requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append({'path': self.path, 'headers': dict(self.headers), 'body': body})
                reply = json.dumps({'model': model, 'choices': [{'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': 'A blue square.'}}],
                    'usage': {'prompt_tokens': 300, 'completion_tokens': 20, 'total_tokens': 320,
                              'completion_tokens_details': {'reasoning_tokens': 12}}}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}'
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    running = Server()
    yield running
    running.close()


def local(tmp_path, url='http://127.0.0.1:8000'):
    return replace(configuration(tmp_path), perception_url=url, perception_model=SERVED)


def test_router_selection(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert router_for(configuration(tmp_path)) is None  # No key: listed but not configured, as before.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'not-exposed')
    gemini = router_for(configuration(tmp_path))
    assert not gemini.local and gemini.model == MODEL and gemini.label == 'Gemini 3.8 Flash'
    served = router_for(local(tmp_path))
    assert served.local and served.model == SERVED and served.endpoint == 'http://127.0.0.1:8000/v1/chat/completions'
    assert SERVED in served.label
    assert router_for(replace(configuration(tmp_path), external_perception=False)) is None


@pytest.mark.parametrize('url', ['http://h:8000', 'http://h:8000/', 'http://h:8000/v1', 'http://h:8000/v1/'])
def test_chat_endpoint(url):
    assert chat_endpoint(url) == 'http://h:8000/v1/chat/completions'


@pytest.mark.parametrize('changes', [
    {'perception_url': 'http://127.0.0.1:8000'},
    {'perception_model': SERVED},
    {'perception_url': 'ftp://127.0.0.1:8000', 'perception_model': SERVED},
    {'perception_url': 'http://user:secret@127.0.0.1:8000', 'perception_model': SERVED},
    {'perception_url': 'http://127.0.0.1:8000', 'perception_model': 'two words'},
    {'perception_url': 'http://127.0.0.1:8000', 'perception_model': SERVED, 'external_perception': False},
])
def test_invalid_perception_settings(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(configuration(tmp_path), **changes)


def test_sessions_saved_before_the_setting_keep_gemini():
    value = retire_fields({'external_perception': True})
    assert value['perception_url'] is None and value['perception_model'] is None


def test_local_payload_names_the_served_model_without_openrouter_routing():
    payload, _ = GeminiRouter(endpoint='http://h:1/v1/chat/completions', model=SERVED).prepare(
        {'messages': QUESTION, 'reasoning': {'effort': 'low'}})
    assert payload['model'] == SERVED and 'provider' not in payload
    assert payload['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert GeminiRouter('key').prepare({'messages': QUESTION})[0]['provider']['allow_fallbacks'] is False


def test_local_reply_checks_the_model_and_costs_zero():
    reply = {'model': SERVED, 'choices': [{'finish_reason': 'stop', 'message': {'content': 'x'}}],
             'usage': {'total_tokens': 10}}
    router = GeminiRouter(endpoint='http://h:1/v1/chat/completions', model=SERVED, send=lambda payload: reply)
    result, tokens = router.execute(*router.prepare({'messages': QUESTION}))
    assert tokens == 10 and result['model'] == SERVED
    assert result['cost_usd'] == 0.0 and result['usage']['cost'] == 0.0
    other = GeminiRouter(endpoint='http://h:1/v1/chat/completions', model=SERVED,
                         send=lambda payload: {**reply, 'model': 'something-else'})
    with pytest.raises(ValueError, match='unexpected model'):
        other.execute(*other.prepare({'messages': QUESTION}))


def test_round_trip_through_the_session_ledger(tmp_path, server, monkeypatch):
    # The host's web proxy is for the internet; a session's own server is reached directly.
    for name in ('http_proxy', 'HTTP_PROXY'):
        monkeypatch.setenv(name, 'http://127.0.0.1:9')
    for name in ('no_proxy', 'NO_PROXY'):
        monkeypatch.delenv(name, raising=False)
    from services.controller.supervisor import Supervisor
    config = local(tmp_path, server.url)
    s = Supervisor(config, lambda: Backend(), router_for(config), runner_factory=Runner)
    try:
        gateway, token = s.start_interactive()
        tool = {t['name']: t for t in gateway.definitions()}['gemini_generate']
        assert SERVED in tool['description'] and 'Gemini 3.8 Flash' not in tool['description']
        result = value(gateway.handle(token, packet('gemini_generate', messages=QUESTION)))
        assert result['message']['content'] == 'A blue square.' and result['cost_usd'] == 0.0
        budget = s.budget()
        assert budget['calls_used'] == 1 and budget['tokens_reported'] == 320
        assert budget['session_cost_reserved_usd'] == 0 and budget['session_cost_reported_usd'] == 0
        assert budget['session_cost_blocked'] is False
        s.finish_interactive()
    finally:
        s.close()
    [request] = server.requests
    assert request['path'] == '/v1/chat/completions'
    assert 'Authorization' not in request['headers']
    assert request['body']['model'] == SERVED and 'provider' not in request['body']


def test_default_description_is_unchanged():
    assert gemini_description(None) == gemini_description('Gemini 3.8 Flash') == GEMINI_DESCRIPTION
    assert GEMINI_DESCRIPTION.startswith('OpenRouter-style chat completion via Gemini 3.8 Flash')
    gateway = Gateway(Backend(), None, audit=None, charge=None, refund=None, budget=None)
    tool = {t['name']: t for t in gateway.definitions()}['gemini_generate']
    assert tool['description'] == GEMINI_DESCRIPTION


@pytest.mark.parametrize('cli', ['claude', 'codex'])
def test_workspace_names_the_served_model(tmp_path, cli):
    setup = Setup(Contract(), task='make_kong', cli=cli, perception_model=SERVED)
    build(recipe_for(), setup, tmp_path)
    skill = (tmp_path/'.agents/skills/gemini/SKILL.md').read_text()
    api = (tmp_path/'.agents/skills/gemini/references/api.md').read_text()
    assert f'served by `{SERVED}`' in skill and f'Use {SERVED} (an open-weight model) through MCP' in skill
    assert 'through OpenRouter' not in skill and 'effort": "low"' not in skill and 'max_tokens=1024)' in skill
    assert f'served by `{SERVED}`' in api and 'google/gemini-3.8-flash' not in api
    assert '{%' not in skill + api and '{{' not in skill + api
