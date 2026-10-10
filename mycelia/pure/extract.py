"""Deterministic extraction from Sphinx documentation HTML (docs.python.org).

For each `str.<method>` entry: signature, description sentences and doctest
examples (`>>> expr` followed by its printed output). Pure text processing;
the page is untrusted data and nothing in it is executed here.
"""
from __future__ import annotations
import html, re

ENTRY = re.compile(r'<dt[^>]*\sid="str\.(\w+)"[^>]*>(.*?)</dt>', re.S)
NEXT_DT = re.compile(r'<dt[^>]*\sid="[\w.]+"', re.S)


def text(fragment):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', fragment))).strip()


def examples(fragment):
    out = []
    for pre in re.findall(r'<pre>(.*?)</pre>', fragment, re.S):
        lines = html.unescape(re.sub(r'<[^>]+>', '', pre)).splitlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('>>> '):
                src, j, result = lines[i][4:].strip(), i + 1, []
                while j < len(lines) and lines[j].strip() and not lines[j].startswith(('>>> ', '... ')):
                    result.append(lines[j]); j += 1
                out.append({'source': src, 'expected': '\n'.join(result).strip()}); i = j
            else:
                i += 1
    return out


def params(signature):
    m = re.search(r'\((.*)\)', signature)
    if not m:
        return []
    ps = [p.strip() for p in m.group(1).split(',') if p.strip() and p.strip() not in ('/', '*', '[', ']')]
    return [{'name': p.split('=')[0].strip('[] '), 'optional': '=' in p or '[' in p} for p in ps]


def extract(page_html, url):
    entries, matches = [], list(ENTRY.finditer(page_html))
    for k, m in enumerate(matches):
        start = m.end(); nxt = NEXT_DT.search(page_html, start)
        body = page_html[start:nxt.start() if nxt else len(page_html)]
        signature = text(m.group(2)).rstrip('¶').strip()
        paragraphs = [text(p) for p in re.findall(r'<p>(.*?)</p>', body, re.S)]
        entries.append({'method': m.group(1), 'signature': signature, 'params': params(signature),
                        'description': ' '.join(p for p in paragraphs if p), 'examples': examples(body),
                        'source': url, 'position': m.start()})
    return entries
