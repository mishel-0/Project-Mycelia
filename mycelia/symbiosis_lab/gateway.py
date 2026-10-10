"""The lab's only door to the outside: a small set of explicit, logged tools.

Retrieved content is UNTRUSTED DATA. It is returned as text tagged
untrusted=True and is never executed, parsed for instructions, or used to
change permissions, policy or code.
"""
from __future__ import annotations
import html, json, os, re, resource, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
from urllib.parse import urlparse
from .registry import AppendLog

DEFAULT_ALLOWLIST = ('arxiv.org', 'export.arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'eutils.ncbi.nlm.nih.gov',
                     'www.ncbi.nlm.nih.gov', 'docs.python.org', 'numpy.org')
SECRET_ENV = re.compile(r'(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PROXY)', re.I)


class GatewayError(Exception):
    pass


class Gateway:
    def __init__(self, root, allowlist=DEFAULT_ALLOWLIST, documents=(), datasets=None, timeout=10.,
                 max_bytes=2_000_000, min_interval=1., opener=None, release_fn=None):
        self.log = AppendLog(Path(root) / 'gateway.jsonl'); self.allowlist = tuple(allowlist)
        self.documents = {Path(p).name: Path(p) for p in documents}; self.datasets = datasets or {}
        self.timeout, self.max_bytes, self.min_interval = timeout, max_bytes, min_interval
        self.opener = opener or urllib.request.urlopen; self.release_fn = release_fn; self._last = 0.; self._raw = False

    def _record(self, tool, arg, ok, detail=''):
        self.log.append({'tool': tool, 'arg': str(arg)[:300], 'ok': ok, 'detail': str(detail)[:300]})

    def fetch_page(self, url):
        u = urlparse(url)
        if u.scheme != 'https' or u.hostname not in self.allowlist:
            self._record('fetch_page', url, False, 'domain or scheme not allowed'); raise GatewayError('not allowlisted: ' + url)
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        try:
            req = urllib.request.Request(url, method='GET', headers={'User-Agent': 'mycelia-lab/0.1'})
            with self.opener(req, timeout=self.timeout) as r:
                raw = r.read(self.max_bytes + 1)
        except Exception as e:  # network refusal, timeout, HTTP error
            self._record('fetch_page', url, False, repr(e)); raise GatewayError(f'fetch failed: {e!r}') from e
        if len(raw) > self.max_bytes:
            self._record('fetch_page', url, False, 'too large'); raise GatewayError('page exceeds size limit')
        if self._raw:
            self._record('fetch_page', url, True, f'{len(raw)} raw bytes')
            return {'url': url, 'html': raw.decode('utf-8', 'replace'), 'untrusted': True, 'retrieved': time.time()}
        text = raw.decode('utf-8', 'replace')
        text = re.sub(r'(?is)<(script|style)[^>]*>.*?</\1>', ' ', text)
        text = html.unescape(re.sub(r'(?s)<[^>]+>', ' ', text)); text = re.sub(r'\s+', ' ', text).strip()
        self._record('fetch_page', url, True, f'{len(text)} chars')
        return {'url': url, 'text': text, 'untrusted': True, 'retrieved': time.time()}

    def fetch_html(self, url):
        """Same checks as fetch_page, but returns the untrusted HTML for structural parsing."""
        self._raw = True
        try:
            return self.fetch_page(url)
        finally:
            self._raw = False

    def read_document(self, source_id):
        if source_id not in self.documents:
            self._record('read_document', source_id, False, 'unknown'); raise GatewayError('unknown document ' + source_id)
        text = self.documents[source_id].read_text()[:self.max_bytes]
        self._record('read_document', source_id, True, f'{len(text)} chars')
        return {'source': source_id, 'text': text, 'untrusted': True, 'retrieved': time.time()}

    def query_approved_dataset(self, dataset_id):
        if dataset_id not in self.datasets:
            self._record('query_approved_dataset', dataset_id, False, 'not approved'); raise GatewayError('dataset not approved')
        self._record('query_approved_dataset', dataset_id, True); return self.datasets[dataset_id]

    def run_sandboxed_experiment(self, spec, cpu_seconds=600, memory_bytes=6 << 30, wall_seconds=900):
        """Run a candidate scorecard in a child process with no network and no secrets."""
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
        code_root = str(Path(__file__).resolve().parents[2])
        boot = (f'import sys, runpy; sys.path.insert(0, {code_root!r}); '
                "runpy.run_module('mycelia.symbiosis_lab.worker', run_name='__main__')")
        def limits():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / 'spec.json').write_text(json.dumps(spec))
            try:
                p = subprocess.run([sys.executable, '-I', '-c', boot, 'spec.json', 'out.json'],
                                   cwd=work, env=env, preexec_fn=limits, capture_output=True, text=True, timeout=wall_seconds)
            except subprocess.TimeoutExpired:
                self._record('run_sandboxed_experiment', spec.get('id'), False, 'wall timeout'); raise GatewayError('experiment timed out')
            if p.returncode:
                self._record('run_sandboxed_experiment', spec.get('id'), False, p.stderr[-300:])
                raise GatewayError('experiment failed: ' + p.stderr[-500:])
            out = json.loads((Path(work) / 'out.json').read_text())
        self._record('run_sandboxed_experiment', spec.get('id'), True); return out

    def submit_candidate(self, candidate_id):
        if self.release_fn is None:
            raise GatewayError('no release gate connected')
        self._record('submit_candidate', candidate_id, True); return self.release_fn(candidate_id)
