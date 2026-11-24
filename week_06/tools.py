"""Real HTTP tools; cache keys depend only on queries/URLs, never gold answers."""
import hashlib
import ipaddress
import json
import os
import socket
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


def request(url, payload=None, headers=None, timeout=40):
    hdr = {'User-Agent': 'AgentFlowCourseProject/1.0 (educational QA)', **(headers or {})}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        hdr['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=hdr)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(2_000_000).decode('utf-8', errors='replace')
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    # Do not echo API keys or credential-bearing request headers into logs.
    raise RuntimeError(f'HTTP request failed ({type(last).__name__})') from None


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def public_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password:
        raise ValueError('read requires a public http(s) URL without credentials')
    for addr in socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80)):
        if not ipaddress.ip_address(addr[4][0]).is_global:
            raise ValueError('Private/local addresses are not allowed')
    return url


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Tools:
    def __init__(self, backend='wiki', cache_dir='.cache/tools', top_k=3):
        self.backend, self.top_k = backend, top_k
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if backend == 'serper' and not os.getenv('SERPER_API_KEY'):
            raise ValueError('Set SERPER_API_KEY before selecting serper')

    def wiki(self, query):
        params = urllib.parse.urlencode({'action': 'query', 'format': 'json', 'generator': 'search',
            'gsrsearch': query, 'gsrlimit': self.top_k, 'prop': 'extracts|info', 'inprop': 'url',
            'explaintext': 1, 'exintro': 1, 'exchars': 2200})
        obj = json.loads(request('https://en.wikipedia.org/w/api.php?' + params))
        if 'error' in obj:
            raise RuntimeError('Wikipedia API returned an error')
        pages = obj.get('query', {}).get('pages', {})
        return [{'title': p.get('title'), 'url': p.get('fullurl'), 'text': p.get('extract', '')}
                for p in sorted(pages.values(), key=lambda p: p.get('index', 999))]

    def search(self, query):
        if self.backend == 'wiki':
            return self.wiki(query)
        obj = json.loads(request('https://google.serper.dev/search',
            {'q': query, 'num': self.top_k}, {'X-API-KEY': os.environ['SERPER_API_KEY']}))
        if 'organic' not in obj:
            raise RuntimeError('Serper response lacks organic results; check quota/key')
        return [{'title': x.get('title'), 'url': x.get('link'), 'text': x.get('snippet', '')}
                for x in obj['organic'][:self.top_k]]

    def read(self, url):
        public_url(url)
        opener = urllib.request.build_opener(SafeRedirect())
        req = urllib.request.Request(url, headers={'User-Agent': 'AgentFlowCourseProject/1.0'})
        with opener.open(req, timeout=40) as r:
            content_type = r.headers.get('Content-Type', '')
            if not any(t in content_type for t in ('text/', 'json', 'xml')):
                raise ValueError('Only text/HTML pages supported')
            html = r.read(1_000_000).decode('utf-8', errors='replace')
        parser = TextExtractor()
        parser.feed(html)
        return {'url': url, 'text': ' '.join(parser.parts)[:7000]}

    def execute(self, action):
        try:
            tool = action.get('tool')
            if tool not in ('search', 'wiki', 'read'):
                return 'ToolError: unknown or invalid tool; choose search, wiki, read, reason, finish'
            arg = action.get('url' if tool == 'read' else 'query')
            if not isinstance(arg, str) or not arg.strip():
                return 'ToolError: missing nonempty query/url'
            cache_key = json.dumps([self.backend, self.top_k, tool, arg], ensure_ascii=False)
            path = self.cache_dir / (hashlib.sha256(cache_key.encode()).hexdigest() + '.json')
            if path.exists():
                return path.read_text(encoding='utf-8')
            result = getattr(self, tool)(arg)
            text = json.dumps(result, ensure_ascii=False)
            path.write_text(text, encoding='utf-8')
            return text
        except Exception as e:
            # Failures are observations so the planner can recover. Do not cache failures.
            return f'ToolError: {type(e).__name__}: {e}'
