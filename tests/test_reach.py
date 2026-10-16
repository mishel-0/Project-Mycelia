import pytest
from mycelia.pure.extract import extract_rst
from mycelia.reach.channels import Reach, ChannelError, GitHubChannel, WebChannel
from mycelia.symbiosis_lab.registry import AppendLog

RST = """.. method:: str.upper()

   Return a copy of the string with all the cased characters converted to
   uppercase.  For example, ``'ab'.upper()`` returns ``'AB'``.

.. method:: str.split(sep=None, maxsplit=-1)

   Return a list of the words in the string, using *sep* as the delimiter.

   .. doctest::

      >>> '1,2,3'.split(',')
      ['1', '2', '3']

.. method:: str.zfill(width)

   Return a copy of the string left filled with ASCII ``'0'`` digits.
"""


def test_rst_extraction_doctests_inline_examples_and_params():
    e = {x['method']: x for x in extract_rst(RST, 'github://python/cpython/Doc/builtins/stdtypes.rst')}
    assert e['upper']['examples'] == [{'source': "'ab'.upper()", 'expected': "'AB'"}]
    assert e['split']['examples'] == [{'source': "'1,2,3'.split(',')", 'expected': "['1', '2', '3']"}]
    assert e['zfill']['params'] == [{'name': 'width', 'optional': False}] and 'delimiter' in e['split']['description']


def test_routing_and_refusals(tmp_path):
    r = Reach(tmp_path)
    assert GitHubChannel(AppendLog(tmp_path / 'l')).can_handle('github://python/cpython/Doc/x.rst')
    with pytest.raises(ChannelError):
        r.read('https://evil.example.com/page')       # not on the web allowlist
    with pytest.raises(ChannelError):
        r.read('ftp://example.com/file')              # no channel handles it
    assert r.log.verify() and any(not x['ok'] for x in r.log.records())
    assert set(r.doctor()) == {'github', 'rss', 'web'}


def test_web_channel_falls_back_then_reports_every_backend(tmp_path, monkeypatch):
    w = WebChannel(AppendLog(tmp_path / 'l'), allowlist={'docs.python.org'})
    monkeypatch.setattr(w, '_get', lambda url: (_ for _ in ()).throw(OSError('blocked ' + url[:20])))
    with pytest.raises(ChannelError) as e:
        w.read('https://docs.python.org/3/')
    assert 'direct' in str(e.value) and 'jina' in str(e.value)
