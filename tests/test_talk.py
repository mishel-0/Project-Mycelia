import json
from types import SimpleNamespace as NS
from mycelia.talk.commands import Mycelia
from mycelia.talk.llm import ClaudeChat, TOOLS


def test_commands_read_real_lab_records():
    m = Mycelia(bundle_path='missing.pkl')
    assert m.weakness()['weak_class'] == 'glioma'
    st = m.status(); assert st['production']['id'] == 'baseline' and st['awaiting_human_approval']
    assert all('status' in c for c in m.history(3)['candidates'])
    assert 'error' in m.why() and 'error' in m.diagnose('no/such.jpg')
    assert 'Unknown command' in m.run('fly') and 'diagnose' in m.run('help')


class FakeClient:
    """Stand-in for anthropic.Anthropic: asks for one tool, then answers from its result."""
    def __init__(self):
        self.calls = []; self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return NS(stop_reason='tool_use', content=[NS(type='tool_use', id='t1', name='lab_weakness', input={})])
        result = json.loads(kw['messages'][-1]['content'][0]['content'])
        return NS(stop_reason='end_turn', content=[NS(type='text', text=f"Weakest class: {result['weak_class']}")])


def test_claude_answers_only_through_mycelia_tools():
    fake = FakeClient(); chat = ClaudeChat(Mycelia(bundle_path='missing.pkl'), client=fake)
    assert chat.send('What is Mycelia worst at?') == 'Weakest class: glioma'
    first = fake.calls[0]
    assert first['model'] == 'claude-opus-5-5' and first['fallbacks'] == 'default' and first['thinking'] == {'type': 'adaptive'}
    assert {t['name'] for t in first['tools']} == {t['name'] for t in TOOLS}
    assert all(t['strict'] and t['input_schema']['additionalProperties'] is False for t in TOOLS)
    tool_msg = chat.messages[2]['content'][0]
    assert tool_msg['type'] == 'tool_result' and tool_msg['tool_use_id'] == 't1' and not tool_msg['is_error']


def test_tool_errors_are_reported_not_raised():
    chat = ClaudeChat(Mycelia(bundle_path='missing.pkl'), client=FakeClient())
    content, err = chat._call_tool('diagnose_scan', {'path': 'nope.jpg'}); assert err and 'not found' in content
    content, err = chat._call_tool('rm_rf', {}); assert err
