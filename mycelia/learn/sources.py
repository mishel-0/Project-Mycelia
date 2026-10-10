"""Source adapters: rst documentation, Python code docstrings, markdown articles.

Every adapter returns Claim dicts with the same shape:
  {subject, relation: example|describes|signature, object, domain, source, lineage, position}
`subject` is an operation name such as 'str.split', 'sorted' or 'statistics.median'.
Text is untrusted: it is parsed, never executed (examples run later only in the sandbox).
"""
from __future__ import annotations
import ast, re
from dataclasses import dataclass
from mycelia.pure.extract import params, _plain, INLINE_EXAMPLE


@dataclass(frozen=True)
class Source:
    id: str
    uri: str          # e.g. github://python/cpython/Doc/library/math.rst
    format: str       # rst | code | markdown
    lineage: str      # who published it (independence of evidence), e.g. 'python/cpython'
    domain: str


def subject_of(expr):
    """Operation called first in an example expression, or None."""
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError:
        return None
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    for c in reversed(calls):  # innermost call first: '1,2'.split(',') -> str.split
        f = c.func
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Constant) and isinstance(f.value.value, str):
                return 'str.' + f.attr
            if isinstance(f.value, ast.Name):
                return f'{f.value.id}.{f.attr}'
    return None


def _claims(subject, sig, description, examples, source, position):
    out = [{'subject': subject, 'relation': 'signature', 'object': {'signature': sig, 'params': params(sig)}},
           {'subject': subject, 'relation': 'describes', 'object': description}]
    out += [{'subject': subject, 'relation': 'example', 'object': ex} for ex in examples]
    for c in out:
        c.update(domain=source.domain, source=source.id, lineage=source.lineage, position=position)
    return out


def qualify(expr, module, names):
    """`median([1, 3])` -> `statistics.median([1, 3])` for functions documented in this module."""
    for n in names:
        expr = re.sub(rf'(?<![\w.]){re.escape(n)}\(', f'{module}.{n}(', expr)
    return expr


def from_rst(text, source):
    """`.. method:: str.x(...)` and `.. function:: name(...)` entries (module from `.. module::`).
    Examples that call this module's functions bare are qualified with the module name."""
    claims = _from_rst(text, source)
    by_module = {}
    for c in claims:
        if '.' in c['subject'] and not c['subject'].startswith('str.'):
            mod, name = c['subject'].rsplit('.', 1); by_module.setdefault(mod, set()).add(name)
    for c in claims:
        if c['relation'] == 'example':
            for mod, names in by_module.items():
                c['object'] = dict(c['object'], source=qualify(c['object']['source'], mod, sorted(names, key=len, reverse=True)))
    return claims


def _from_rst(text, source):
    lines, claims, module, i = text.splitlines(), [], '', 0
    while i < len(lines):
        mod = re.match(r'\.\. (?:module|currentmodule):: ([\w.]+)', lines[i])
        if mod:
            module = mod.group(1)
        m = re.match(r'\.\. (method|function|class):: ([\w.]+)(\(.*\))?\s*$', lines[i])  # classes like int(...) are callables
        if not m:
            i += 1; continue
        name, sig = m.group(2), m.group(3) or '()'
        subject = name if (m.group(1) == 'method' or '.' in name or not module or module == 'builtins') else f'{module}.{name}'
        j, body = i + 1, []
        while j < len(lines) and (not lines[j].strip() or lines[j].startswith('   ')):
            body.append(lines[j][3:] if lines[j].startswith('   ') else ''); j += 1
        exs, prose, k = [], [], 0
        while k < len(body):
            s = body[k].strip()
            if s.startswith('>>> '):
                src, k, out = s[4:], k + 1, []
                while k < len(body) and body[k].strip() and not body[k].strip().startswith(('>>> ', '... ')):
                    out.append(body[k].strip()); k += 1
                exs.append({'source': src, 'expected': '\n'.join(out)}); continue
            if s and not s.startswith(('..', ':')):
                prose.append(s)
            k += 1
        joined = ' '.join(prose)
        exs += [{'source': a, 'expected': b} for a, b in INLINE_EXAMPLE.findall(joined)]
        claims += _claims(subject, f'{subject}{sig}', _plain(joined), exs, source, i)
        i = j
    return claims


def from_code(text, source, module):
    """Doctests in the docstrings of a module's public top-level functions (AST only, never run here)."""
    claims = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            doc = ast.get_docstring(node) or ''
            lines, exs, k = doc.splitlines(), [], 0
            while k < len(lines):
                s = lines[k].strip()
                if s.startswith('>>> '):
                    src, k, out = s[4:], k + 1, []
                    while k < len(lines) and lines[k].strip() and not lines[k].strip().startswith(('>>> ', '... ')):
                        out.append(lines[k].strip()); k += 1
                    sub = subject_of(src)
                    exs.append({'source': src if not sub or sub == node.name else src, 'expected': '\n'.join(out)}); continue
                k += 1
            # Doctests call `median(...)`; qualify bare calls with the module so they name the same operation.
            exs = [{'source': re.sub(rf'(?<![\w.]){node.name}\(', f'{module}.{node.name}(', e['source']), 'expected': e['expected']}
                   for e in exs]
            args = ', '.join(a.arg for a in node.args.args)
            first = doc.split('\n\n')[0].replace('\n', ' ')
            claims += _claims(f'{module}.{node.name}', f'{module}.{node.name}({args})', first, exs, source, node.lineno)
    return claims


CHEAT = re.compile(r'^<[\w ]+>\s*=\s*(.+?)\s{2,}#\s*(.+)$')


def from_markdown(text, source):
    """Fenced code blocks: one-line doctests, and annotated lines like
    `<str> = <str>.strip()   # Strips all whitespace characters from both ends.`"""
    claims = []
    for block in re.findall(r'```(?:python|py)?\n(.*?)```', text, re.S):
        lines = block.splitlines()
        for k, line in enumerate(lines):
            s = line.strip()
            if s.startswith('>>> ') and k + 1 < len(lines) and not lines[k + 1].startswith(('>>> ', '... ')):
                sub = subject_of(s[4:])
                if sub:
                    claims += [dict(subject=sub, relation='example', object={'source': s[4:], 'expected': lines[k + 1].strip()},
                                    domain=source.domain, source=source.id, lineage=source.lineage, position=k)]
            m = CHEAT.match(s)
            if m:
                expr = re.sub(r'<(\w+)>', 'x', m.group(1))
                sub = subject_of(expr)
                if sub and sub.startswith('x.'):
                    sub = 'str.' + sub[2:]  # <str>.strip() annotated with prose
                if sub:
                    claims.append(dict(subject=sub, relation='describes', object=m.group(2).strip(), domain=source.domain,
                                       source=source.id, lineage=source.lineage, position=k))
    return claims


def extract(text, source):
    if source.format == 'rst':
        return from_rst(text, source)
    if source.format == 'code':
        return from_code(text, source, source.uri.rsplit('/', 1)[-1].removesuffix('.py'))
    if source.format == 'markdown':
        return from_markdown(text, source)
    raise ValueError(source.format)
