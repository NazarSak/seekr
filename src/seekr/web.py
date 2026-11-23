"""A tiny web UI (stdlib http.server) in deliberately plain 1998 style.

Routes
  /                 search page (works without JavaScript; JS adds live results)
  /search?q=        server-rendered results
  /api/search?q=    JSON
  /api/suggest?q=   JSON autocomplete (top 5 terms by document frequency)
  /doc?id=          full text of one document
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .engine import SearchEngine
from .snippets import HTML

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ background:#fff; color:#000; font-family:"Times New Roman",serif; margin:2em auto; max-width:46em; padding:0 1em; }}
 h1 {{ font-size:2.2em; margin:0; }} h1 span {{ color:#00c; }}
 .tag {{ color:#666; font-size:.9em; margin-bottom:1em; }}
 input[type=text] {{ width:26em; font-size:1em; }}
 a {{ color:#00c; }} a:visited {{ color:#551a8b; }}
 .hit {{ margin:1em 0; }} .hit .t {{ font-size:1.1em; }}
 .meta {{ color:#060; font-size:.85em; }} .snip {{ margin:.2em 0; }}
 .why {{ background:#ffc; border:1px solid #cc9; font:12px monospace; padding:4px; display:inline-block; }}
 hr {{ border:0; border-top:1px solid #999; }} pre {{ white-space:pre-wrap; }}
 #sug {{ color:#666; font-size:.85em; }}
</style></head><body>
<h1><a href="/" style="text-decoration:none;color:#000"><span>S</span>eekr</a></h1>
<div class="tag">a small search engine &middot; {n_docs} documents &middot; {n_terms} terms indexed</div>
<form action="/search" method="get">
<input type="text" name="q" id="q" value="{q}" autocomplete="off" autofocus>
<input type="submit" value="Search">
<label><input type="checkbox" name="why" value="1" {why}> explain scores</label>
<div id="sug"></div>
</form>
<hr>
<div id="out">{body}</div>
<hr><div class="tag">Try: <a href="/search?q=%22search+intent%22">"search intent"</a> &middot;
<a href="/search?q=crawl+NOT+budget">crawl NOT budget</a> &middot;
<a href="/search?q=canonicl~">canonicl~</a> &middot;
<a href="/search?q=schem*">schem*</a></div>
<script>
 var q=document.getElementById('q'),out=document.getElementById('out'),sug=document.getElementById('sug'),t;
 function esc(s){{return s.replace(/[&<>"]/g,function(c){{return {{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]}})}}
 q.addEventListener('input',function(){{
   clearTimeout(t);
   t=setTimeout(function(){{
     var v=q.value.trim(); if(!v){{out.innerHTML='';sug.textContent='';return;}}
     var last=v.split(/\\s+/).pop();
     fetch('/api/suggest?q='+encodeURIComponent(last)).then(function(r){{return r.json()}})
       .then(function(a){{sug.textContent=a.length?'suggestions: '+a.join(', '):''}});
     fetch('/api/search?q='+encodeURIComponent(v)).then(function(r){{return r.json()}}).then(function(d){{
       var h='<div class="meta">'+d.total+' result(s) in '+d.elapsed_ms.toFixed(2)+' ms'+(d.note?' &mdash; '+esc(d.note):'')+'</div>';
       d.hits.forEach(function(x){{h+='<div class="hit"><a class="t" href="/doc?id='+x.id+'">'+esc(x.title)+'</a>'+
         '<div class="snip">'+x.snippet+'</div><div class="meta">'+esc(x.path)+' &middot; score '+x.score.toFixed(3)+'</div></div>'}});
       out.innerHTML=h;
     }});
   }},120);
 }});
</script>
</body></html>"""


def _explain_html(hit) -> str:
    e = hit.explanation
    if not e:
        return ""
    rows = "<br>".join(
        f"{html.escape(p.term)}: tf={p.tf} df={p.df} idf={p.idf:.3f} &rarr; {p.weight:.4f}" for p in e.parts
    )
    return f'<div class="why">{e.ranker}: {rows}<br>norm={e.norm:.3f} &rarr; score={e.score:.4f}</div>'


def render_results(engine: SearchEngine, q: str, why: bool) -> str:
    res = engine.search(q, limit=20, markup=HTML, explain=why)
    note = f" &mdash; {html.escape(res.note)}" if res.note else ""
    out = [f'<div class="meta">{res.total} result(s) in {res.elapsed_ms:.2f} ms{note}</div>']
    for h in res.hits:
        out.append(
            f'<div class="hit"><a class="t" href="/doc?id={h.doc.id}">{html.escape(h.doc.title)}</a>'
            f'<div class="snip">{h.snippet}</div>'
            f'<div class="meta">{html.escape(h.doc.path)} &middot; score {h.score:.3f}</div>{_explain_html(h)}</div>'
        )
    return "\n".join(out)


def make_handler(engine: SearchEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, body: str, ctype="text/html; charset=utf-8", status=200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _page(self, q="", body="", why=False, title="Seekr"):
            return PAGE.format(
                title=html.escape(title),
                q=html.escape(q, quote=True),
                body=body,
                why="checked" if why else "",
                n_docs=engine.index.n_docs,
                n_terms=len(engine.index.postings),
            )

        def do_GET(self):  # noqa: N802
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            q = (qs.get("q", [""])[0]).strip()
            try:
                if u.path == "/":
                    self._send(self._page())
                elif u.path == "/search":
                    why = "why" in qs
                    self._send(self._page(q, render_results(engine, q, why) if q else "", why, f"{q} - Seekr"))
                elif u.path == "/api/search":
                    res = engine.search(q, limit=20, markup=HTML) if q else None
                    payload = {"total": 0, "elapsed_ms": 0.0, "note": "", "hits": []}
                    if res:
                        payload = {
                            "total": res.total,
                            "elapsed_ms": res.elapsed_ms,
                            "note": res.note,
                            "hits": [
                                {"id": h.doc.id, "title": h.doc.title, "path": h.doc.path,
                                 "score": h.score, "snippet": h.snippet}
                                for h in res.hits
                            ],
                        }
                    self._send(json.dumps(payload), "application/json; charset=utf-8")
                elif u.path == "/api/suggest":
                    self._send(json.dumps(engine.suggest(q)), "application/json; charset=utf-8")
                elif u.path == "/doc":
                    doc_id = int(qs.get("id", ["-1"])[0])
                    if not 0 <= doc_id < engine.index.n_docs:
                        self._send(self._page(body="<p>No such document.</p>"), status=404)
                        return
                    d = engine.index.docs[doc_id]
                    text = html.escape(engine.read_doc(doc_id))
                    self._send(self._page(body=f"<h2>{html.escape(d.title)}</h2><pre>{text}</pre>", title=d.title))
                else:
                    self._send(self._page(body="<p>404 - not found.</p>"), status=404)
            except ValueError:
                self._send(self._page(body="<p>Bad request.</p>"), status=400)

    return Handler


def serve(engine: SearchEngine, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(engine))
    print(f"Seekr web UI: http://{host}:{port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()
