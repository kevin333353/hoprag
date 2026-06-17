"""Bake the example cache into a single self-contained HTML file.

Produces demo/standalone.html — open it directly in a browser (double-click / file://)
or host it statically (e.g. GitHub Pages); no backend needed. The ⚡ example questions
replay instantly from the embedded cache; live custom questions require the local server.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    html = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
    data = (ROOT / "demo" / "examples_cache.json").read_text(encoding="utf-8")
    inject = "<script>window.__EXAMPLES__ = " + data + ";</script>\n</head>"
    html = html.replace("</head>", inject, 1)
    out = ROOT / "demo" / "standalone.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
