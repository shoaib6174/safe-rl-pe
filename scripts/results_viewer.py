#!/usr/bin/env python3
"""Simple web-based results viewer. Run on niro-2 to browse GIFs/PNGs/logs remotely."""

import os
import sys
import time
import mimetypes
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from string import Template
from urllib.parse import unquote, quote
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", os.path.expanduser("~/Codes/safe-rl-pe/results")))
PORT = int(os.environ.get("PORT", 8080))

# Custom dashboard sections: list of (section_title, [run_names], description)
DASHBOARD_SECTIONS = [
    (
        "Group 2: Alt-Freeze thr=0.6 (Oscillation)",
        ["SP13e_alt_no_obs_s42", "SP13e_alt_no_obs_s43", "SP13f_alt_obs_s42", "SP13f_alt_obs_s43"],
        "Alternate freeze with 0.6 threshold. Classic oscillation: peaked early (it2-9), stuck in 0.39-0.62 range.",
    ),
    (
        "Group 5: Pure Self-Play",
        ["SP14b_pure_selfplay_s43"],
        "No freeze mechanism. Asymmetric obs, FOV 90°, 8m. Early stage (6 iters).",
    ),
]

HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Results: $title</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { color: #e94560; margin-bottom: 5px; font-size: 1.4em; }
  .breadcrumb { margin-bottom: 20px; color: #888; font-size: 0.9em; }
  .breadcrumb a { color: #0f9; text-decoration: none; }
  .breadcrumb a:hover { text-decoration: underline; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
  .card {
    background: #16213e; border-radius: 8px; overflow: hidden;
    border: 1px solid #0f3460; transition: border-color 0.2s;
  }
  .card:hover { border-color: #e94560; }
  .card a { text-decoration: none; color: inherit; display: block; }
  .card .preview {
    width: 100%; aspect-ratio: 4/3; display: flex; align-items: center;
    justify-content: center; background: #0f0f23; overflow: hidden;
  }
  .card .preview img { max-width: 100%; max-height: 100%; object-fit: contain; }
  .card .icon { font-size: 3em; color: #555; }
  .card .label {
    padding: 10px 12px; font-size: 0.85em; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; border-top: 1px solid #0f3460;
  }
  .card .label .size { color: #888; font-size: 0.8em; float: right; }
  .dirs { margin-bottom: 24px; }
  .dir-table { width: 100%; border-collapse: collapse; }
  .dir-table th {
    text-align: left; padding: 8px 12px; color: #888; font-size: 0.75em;
    text-transform: uppercase; border-bottom: 1px solid #0f3460; cursor: pointer;
    user-select: none;
  }
  .dir-table th:hover { color: #e94560; }
  .dir-table th .arrow { font-size: 0.8em; margin-left: 4px; }
  .dir-table td { padding: 8px 12px; border-bottom: 1px solid #0f346033; font-size: 0.9em; }
  .dir-table tr:hover { background: #16213e; }
  .dir-table a { color: #0f9; text-decoration: none; }
  .dir-table a:hover { color: #e94560; }
  .dir-table .meta { color: #888; font-size: 0.85em; }
  .section-title { color: #888; font-size: 0.8em; text-transform: uppercase; margin: 16px 0 8px; }
  .dash-link { display: inline-block; margin: 8px 0; padding: 8px 16px; background: #e94560; color: #fff; text-decoration: none; border-radius: 6px; font-size: 0.9em; }
  .dash-link:hover { background: #c73652; }
  .batch { margin-bottom: 32px; }
  .batch-title { color: #e94560; font-size: 1.1em; margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #0f3460; }
  .batch-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .batch-card { background: #16213e; border-radius: 8px; overflow: hidden; border: 1px solid #0f3460; }
  .batch-card img { width: 100%; display: block; }
  .batch-card .batch-label { padding: 6px 10px; font-size: 0.8em; text-align: center; border-top: 1px solid #0f3460; }
  .batch-divider { border: none; border-top: 2px solid #0f3460; margin: 24px 0; }
  .section { margin-bottom: 40px; }
  .section-header { color: #e94560; font-size: 1.3em; margin-bottom: 4px; padding-bottom: 8px; border-bottom: 2px solid #e94560; }
  .section-desc { color: #888; font-size: 0.85em; margin-bottom: 16px; }
  .run-group { margin-bottom: 24px; }
  .run-name { color: #0f9; font-size: 0.95em; margin-bottom: 8px; font-weight: bold; }
  .media-row { display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap; }
  .media-card { background: #16213e; border-radius: 8px; overflow: hidden; border: 1px solid #0f3460; flex: 1; min-width: 300px; max-width: 600px; }
  .media-card img { width: 100%; display: block; }
  .media-card .media-label { padding: 6px 10px; font-size: 0.8em; text-align: center; border-top: 1px solid #0f3460; color: #aaa; }
  .nav-links { margin-bottom: 20px; }
  .nav-links a { display: inline-block; margin-right: 12px; padding: 6px 14px; background: #16213e; color: #0f9; text-decoration: none; border-radius: 6px; border: 1px solid #0f3460; font-size: 0.85em; }
  .nav-links a:hover { border-color: #e94560; color: #e94560; }
  .nav-links a.active { background: #e94560; color: #fff; border-color: #e94560; }
</style>
</head>
<body>
<h1>Results Viewer</h1>
$breadcrumb
$content
<script>
function sortTable(colIdx, type) {
  const table = document.querySelector('.dir-table');
  if (!table) return;
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const headers = table.querySelectorAll('th');
  const current = table.dataset.sortCol;
  const currentDir = table.dataset.sortDir || 'asc';
  let dir = (current == colIdx && currentDir === 'asc') ? 'desc' : 'asc';
  table.dataset.sortCol = colIdx;
  table.dataset.sortDir = dir;
  headers.forEach((h, i) => {
    const arrow = h.querySelector('.arrow');
    if (arrow) arrow.textContent = i == colIdx ? (dir === 'asc' ? '▲' : '▼') : '';
  });
  rows.sort((a, b) => {
    let va = a.cells[colIdx].dataset.sort || a.cells[colIdx].textContent;
    let vb = b.cells[colIdx].dataset.sort || b.cells[colIdx].textContent;
    if (type === 'num') { va = parseFloat(va) || 0; vb = parseFloat(vb) || 0; }
    else { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    if (va < vb) return dir === 'asc' ? -1 : 1;
    if (va > vb) return dir === 'asc' ? 1 : -1;
    return 0;
  });
  rows.forEach(r => tbody.appendChild(r));
}
</script>
</body>
</html>""")


def human_size(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def build_breadcrumb(rel_path):
    parts = rel_path.split("/") if rel_path else []
    crumbs = ['<a href="/">root</a>']
    for i, part in enumerate(parts):
        link = "/" + "/".join(parts[: i + 1])
        crumbs.append(f'<a href="{link}">{part}</a>')
    return '<div class="breadcrumb">' + " / ".join(crumbs) + "</div>"


def is_image(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"))


def collect_images_by_batch(results_dir):
    """Collect all images from run directories, grouped by common suffix."""
    from collections import defaultdict
    batches = defaultdict(list)

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        for f in sorted(run_dir.iterdir()):
            if not is_image(f.name):
                continue
            # Extract batch key: strip the run name prefix to find common suffix
            # e.g. "SP12_mc_s42_M3000_trajectories.png" -> "M3000_trajectories.png"
            suffix = f.name
            if suffix.startswith(run_dir.name):
                suffix = suffix[len(run_dir.name):].lstrip("_")
            if not suffix:
                suffix = f.name
            batches[suffix].append((run_dir.name, f))

    return batches


class ResultsHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.strip("/"))

        if path == "dashboard":
            self._serve_dashboard()
            return

        if path == "dashboard/sections":
            self._serve_sections_dashboard()
            return

        full_path = RESULTS_DIR / path

        if not full_path.resolve().is_relative_to(RESULTS_DIR.resolve()):
            self.send_error(403, "Forbidden")
            return

        if full_path.is_file():
            self._serve_file(full_path)
        elif full_path.is_dir():
            self._serve_directory(full_path, path)
        else:
            self.send_error(404, "Not Found")

    def _serve_sections_dashboard(self):
        """Serve dashboard with custom grouped sections showing PNGs and GIFs per run."""
        content_parts = []

        # Navigation
        content_parts.append('<div class="nav-links">')
        content_parts.append('<a href="/dashboard">All Images</a>')
        content_parts.append('<a href="/dashboard/sections" class="active">Group View</a>')
        content_parts.append('</div>')

        for section_title, run_names, description in DASHBOARD_SECTIONS:
            content_parts.append('<div class="section">')
            content_parts.append(f'<div class="section-header">{section_title}</div>')
            content_parts.append(f'<div class="section-desc">{description}</div>')

            for run_name in run_names:
                run_dir = RESULTS_DIR / run_name
                if not run_dir.is_dir():
                    content_parts.append(f'<div class="run-group"><div class="run-name">{run_name} (not found)</div></div>')
                    continue

                content_parts.append(f'<div class="run-group">')
                content_parts.append(f'<div class="run-name">{run_name}</div>')
                content_parts.append('<div class="media-row">')

                # Look for trajectory.png, grid.png, grid.gif
                for fname, label in [("trajectory.png", "Trajectory (1 ep)"), ("grid.png", "Grid PNG (9 ep)"), ("grid.gif", "Grid GIF (animated)")]:
                    fpath = run_dir / fname
                    if fpath.is_file():
                        rel = fpath.relative_to(RESULTS_DIR)
                        link = "/" + quote(str(rel))
                        content_parts.append(
                            f'<div class="media-card">'
                            f'<a href="{link}" target="_blank"><img src="{link}" loading="lazy"></a>'
                            f'<div class="media-label">{label}</div>'
                            f'</div>'
                        )

                content_parts.append('</div></div>')  # close media-row, run-group

            content_parts.append('</div>')  # close section
            content_parts.append('<hr class="batch-divider">')

        breadcrumb = '<div class="breadcrumb"><a href="/">root</a> / <a href="/dashboard">dashboard</a> / sections</div>'
        html = HTML_TEMPLATE.substitute(
            title="Group Sections", breadcrumb=breadcrumb, content="\n".join(content_parts)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_dashboard(self):
        batches = collect_images_by_batch(RESULTS_DIR)
        content_parts = []

        # Navigation
        content_parts.append('<div class="nav-links">')
        content_parts.append('<a href="/dashboard" class="active">All Images</a>')
        content_parts.append('<a href="/dashboard/sections">Group View</a>')
        content_parts.append('</div>')

        # Sort batches: most recent files first
        sorted_batches = sorted(
            batches.items(),
            key=lambda kv: max(f.stat().st_mtime for _, f in kv[1]),
            reverse=True,
        )

        for suffix, items in sorted_batches:
            content_parts.append(f'<div class="batch">')
            content_parts.append(f'<div class="batch-title">{suffix} ({len(items)} runs)</div>')
            content_parts.append('<div class="batch-grid">')
            for run_name, img_path in items:
                rel = img_path.relative_to(RESULTS_DIR)
                link = "/" + quote(str(rel))
                content_parts.append(
                    f'<div class="batch-card">'
                    f'<a href="{link}" target="_blank"><img src="{link}" loading="lazy"></a>'
                    f'<div class="batch-label">{run_name}</div>'
                    f'</div>'
                )
            content_parts.append('</div></div>')
            content_parts.append('<hr class="batch-divider">')

        if not batches:
            content_parts.append('<p style="color:#888;">No images found in run directories.</p>')

        breadcrumb = '<div class="breadcrumb"><a href="/">root</a> / dashboard</div>'
        html = HTML_TEMPLATE.substitute(
            title="Dashboard", breadcrumb=breadcrumb, content="\n".join(content_parts)
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_file(self, full_path):
        mime, _ = mimetypes.guess_type(str(full_path))
        mime = mime or "application/octet-stream"
        try:
            data = full_path.read_bytes()
        except Exception:
            self.send_error(500, "Read error")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "public, max-age=300")
        self.end_headers()
        self.wfile.write(data)

    def _serve_directory(self, full_path, rel_path):
        try:
            entries = sorted(full_path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        except PermissionError:
            self.send_error(403, "Forbidden")
            return

        dirs = []
        images = []
        files = []

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                dirs.append(entry)
            elif is_image(entry.name):
                images.append(entry)
            else:
                files.append(entry)

        content_parts = []

        # Show dashboard link on root page
        if not rel_path:
            content_parts.append('<a class="dash-link" href="/dashboard">View All Images Dashboard</a>')

        if dirs:
            content_parts.append('<div class="section-title">Directories</div><div class="dirs">')
            content_parts.append(
                '<table class="dir-table" data-sort-col="0" data-sort-dir="asc">'
                '<thead><tr>'
                '<th onclick="sortTable(0,\'str\')">Name <span class="arrow">▲</span></th>'
                '<th onclick="sortTable(1,\'num\')">Items <span class="arrow"></span></th>'
                '<th onclick="sortTable(2,\'num\')">Last Modified <span class="arrow"></span></th>'
                '</tr></thead><tbody>'
            )
            for d in dirs:
                link = "/" + (rel_path + "/" + d.name if rel_path else d.name)
                n_items = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
                mtime = d.stat().st_mtime
                mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                content_parts.append(
                    f'<tr>'
                    f'<td><a href="{quote(link)}">{d.name}/</a></td>'
                    f'<td class="meta" data-sort="{n_items}">{n_items}</td>'
                    f'<td class="meta" data-sort="{mtime:.0f}">{mtime_str}</td>'
                    f'</tr>'
                )
            content_parts.append("</tbody></table></div>")

        if images:
            content_parts.append('<div class="section-title">Images & GIFs</div><div class="grid">')
            for img in images:
                link = "/" + (rel_path + "/" + img.name if rel_path else img.name)
                size = human_size(img.stat().st_size)
                content_parts.append(
                    f'<div class="card"><a href="{quote(link)}" target="_blank">'
                    f'<div class="preview"><img src="{quote(link)}" loading="lazy"></div>'
                    f'<div class="label">{img.name}<span class="size">{size}</span></div>'
                    f"</a></div>"
                )
            content_parts.append("</div>")

        if files:
            content_parts.append('<div class="section-title">Files</div><div class="grid">')
            for f in files:
                link = "/" + (rel_path + "/" + f.name if rel_path else f.name)
                size = human_size(f.stat().st_size)
                content_parts.append(
                    f'<div class="card"><a href="{quote(link)}" target="_blank">'
                    f'<div class="preview"><span class="icon">&#128196;</span></div>'
                    f'<div class="label">{f.name}<span class="size">{size}</span></div>'
                    f"</a></div>"
                )
            content_parts.append("</div>")

        if not (dirs or images or files):
            content_parts.append('<p style="color:#888;">Empty directory</p>')

        title = rel_path or "/"
        breadcrumb = build_breadcrumb(rel_path)
        html = HTML_TEMPLATE.substitute(title=title, breadcrumb=breadcrumb, content="\n".join(content_parts))

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


def main():
    if not RESULTS_DIR.is_dir():
        print(f"Error: {RESULTS_DIR} is not a directory")
        sys.exit(1)

    server = HTTPServer(("0.0.0.0", PORT), ResultsHandler)
    print(f"Serving results from: {RESULTS_DIR}")
    print(f"Open in browser: http://100.71.2.97:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
