"""Channels: GitHub (git reads of public repos), Web (direct or Jina Reader), RSS."""
from __future__ import annotations
import hashlib, re, shutil, subprocess, tempfile, time, urllib.request
from pathlib import Path
from mycelia.symbiosis_lab.registry import AppendLog

MAX_BYTES = 5 * 1024 * 1024


class ChannelError(Exception):
    pass


class Channel:
    name, tier, backends = '', 0, []

    def __init__(self, log):
        self.log, self.active_backend = log, None

    def can_handle(self, source):
        raise NotImplementedError

    def record(self, source, ok, detail, sha=None):
        self.log.append({'channel': self.name, 'backend': self.active_backend, 'source': source, 'ok': ok,
                         'detail': str(detail)[:300], 'sha256': sha})

    def result(self, source, text, **meta):
        sha = hashlib.sha256(text.encode()).hexdigest(); self.record(source, True, f'{len(text)} chars', sha)
        return {'source': source, 'text': text, 'sha256': sha, 'untrusted': True, 'retrieved': time.time(),
                'channel': self.name, 'backend': self.active_backend, **meta}


class GitHubChannel(Channel):
    """github://owner/repo/path/to/file — anonymous git read of a public repository (blob-less, depth 1)."""
    name, tier, backends = 'github', 0, ['git']
    PATTERN = re.compile(r'^github://([\w.-]+)/([\w.-]+)/(.+)$')

    def can_handle(self, source):
        return bool(self.PATTERN.match(source))

    def check(self):
        self.active_backend = 'git' if shutil.which('git') else None
        return ('ok', 'git available') if self.active_backend else ('off', 'git not installed')

    def read(self, source, timeout=600):
        m = self.PATTERN.match(source)
        if not m or self.check()[0] != 'ok':
            raise ChannelError('cannot read ' + source)
        owner, repo, path = m.groups()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                subprocess.run(['git', 'clone', '-q', '--depth', '1', '--filter=blob:none', '--no-checkout',
                                f'https://github.com/{owner}/{repo}', tmp], check=True, timeout=timeout, capture_output=True)
                commit = subprocess.run(['git', '-C', tmp, 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
                blob = subprocess.run(['git', '-C', tmp, 'show', f'HEAD:{path}'], check=True, capture_output=True, timeout=timeout).stdout
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.record(source, False, e); raise ChannelError(f'git read failed: {e}')
        if len(blob) > MAX_BYTES:
            self.record(source, False, 'too large'); raise ChannelError('file too large')
        return self.result(source, blob.decode('utf-8', 'replace'), commit=commit, url=f'https://github.com/{owner}/{repo}/blob/{commit}/{path}')


class WebChannel(Channel):
    """https URLs: direct GET, falling back to Jina Reader (Agent-Reach's web backend)."""
    name, tier, backends = 'web', 0, ['direct', 'jina']

    def __init__(self, log, allowlist=None, timeout=20):
        super().__init__(log); self.allowlist, self.timeout = allowlist, timeout

    def can_handle(self, source):
        return source.startswith('https://')

    def _get(self, url):
        req = urllib.request.Request(url, headers={'User-Agent': 'mycelia-reach/0.1', 'Accept': 'text/plain, text/html'})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = r.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ChannelError('response too large')
        return body.decode('utf-8', 'replace')

    def read(self, source):
        host = re.sub(r'^https://([^/]+).*$', r'\1', source)
        if self.allowlist is not None and host not in self.allowlist:
            self.record(source, False, 'host not allowlisted'); raise ChannelError('not allowlisted: ' + host)
        errors = []
        for backend in self.backends:
            self.active_backend = backend
            try:
                return self.result(source, self._get(source if backend == 'direct' else f'https://r.jina.ai/{source}'))
            except Exception as e:
                errors.append(f'{backend}: {e!r}'); self.record(source, False, e)
        self.active_backend = None
        raise ChannelError('; '.join(errors))


class RSSChannel(Channel):
    name, tier, backends = 'rss', 0, ['feedparser']

    def can_handle(self, source):
        return source.startswith('rss+https://')

    def check(self):
        try:
            import feedparser  # noqa: F401
            self.active_backend = 'feedparser'; return 'ok', 'feedparser installed'
        except ImportError:
            return 'off', 'pip install feedparser'

    def read(self, source):
        if self.check()[0] != 'ok':
            raise ChannelError('feedparser not installed')
        import feedparser
        feed = feedparser.parse(source[4:])
        text = '\n\n'.join(f"{e.get('title', '')}\n{e.get('summary', '')}" for e in feed.entries)
        return self.result(source, text)


class Reach:
    """Route a source to the first channel that can handle it."""
    def __init__(self, root, web_allowlist=('docs.python.org',)):
        log = AppendLog(Path(root) / 'reach.jsonl'); self.log = log
        self.channels = [GitHubChannel(log), RSSChannel(log), WebChannel(log, set(web_allowlist))]

    def read(self, source):
        for ch in self.channels:
            if ch.can_handle(source):
                return ch.read(source)
        raise ChannelError('no channel for ' + source)

    def doctor(self):
        out = {}
        for ch in self.channels:
            out[ch.name] = ch.check() if hasattr(ch, 'check') else ('ok', 'checked on read')
        return out
