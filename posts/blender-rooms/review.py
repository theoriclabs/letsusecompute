"""Create a local, anonymous A/B review packet without model identity labels."""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from pathlib import Path


def build_review(source: Path, output: Path):
    raw = (source / "pairs.json").read_bytes()
    pairs = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    (output / "pairs.json").write_bytes(raw)
    cards = []
    for index, pair in enumerate(pairs):
        identity = pair["pair_id"]
        cards.append(
            f'<article data-pair="{html.escape(identity)}"><h2>Room {index + 1}</h2><p class="brief">{html.escape(pair["prompt"])}</p><div class="images">'
        )
        for side in ["A", "B"]:
            name = pair[side]
            cards.append(f"<figure><figcaption>{side}</figcaption>")
            if name:
                image = (source / name).resolve()
                if not image.is_relative_to(source.resolve()):
                    raise ValueError("Review image path escapes packet")
                shutil.copy2(image, output / name)
                cards.append(
                    f'<img src="{html.escape(name)}" width="384" height="384" alt="Room {index + 1}, option {side}" loading="lazy">'
                )
            else:
                cards.append('<div class="missing">No render</div>')
            cards.append("</figure>")
        cards.append("</div><fieldset><legend>Which better meets the brief?</legend>")
        for choice, text in [("A", "A"), ("B", "B"), ("tie", "Tie"), ("neither", "Neither")]:
            cards.append(
                f'<label><input type="radio" name="{identity}" value="{choice}"> {text}</label>'
            )
        cards.append(
            f'</fieldset><label class="notes">Notes (optional)<textarea aria-label="Notes for room {index + 1}" rows="2"></textarea></label></article>'
        )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex"><title>Room comparison — anonymous review</title><style>
body{font:17px/1.5 system-ui,sans-serif;background:#f7f4ef;color:#1e2323;max-width:960px;margin:auto;padding:24px}h1{line-height:1.15}.images{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}figcaption{font-weight:700;padding:8px}img{width:100%;height:auto;display:block}.missing{aspect-ratio:1;background:#e3dfd6;display:grid;place-items:center}article{padding:24px 0;border-top:1px solid #c9c6bd}.brief{font-size:14px;white-space:pre-wrap}fieldset{border:0;padding:12px 0}fieldset label{display:inline-flex;align-items:center;gap:5px;padding:10px 18px 10px 0}input{accent-color:#326e7b}textarea{box-sizing:border-box;display:block;width:100%;font:inherit;border:1px solid #999;padding:8px;border-radius:5px}.notes{display:block;font-size:14px}button{background:#263d42;color:white;border:0;padding:12px 18px;border-radius:5px;cursor:pointer;font:inherit}.toolbar{position:sticky;bottom:0;background:#f7f4efef;padding:16px 0;display:flex;justify-content:space-between;align-items:center;border-top:1px solid #c9c6bd}@media(max-width:600px){body{padding:16px}.images{gap:8px}}
</style></head><body><h1>Compare the rooms</h1><p>Choose the image that better follows each brief. Consider recognizable furniture, detail, layout, and color. A/B labels are shuffled for each pair.</p><p>Your choices stay in this browser. Export your review when finished.</p>"""
    page += (
        "".join(cards)
        + '<div class="toolbar"><span id="progress"></span><button id="export" type="button">Export review</button></div>'
    )
    page += """<script>
const manifest = "DIGEST";
const key = 'blender-review-' + manifest;
const cards = [...document.querySelectorAll('article')];
function answers() { return cards.map(card => ({pair_id: card.dataset.pair, preference: card.querySelector('input:checked')?.value ?? null, notes: card.querySelector('textarea').value})); }
function update() { const rows = answers(); document.querySelector('#progress').textContent = rows.filter(r => r.preference).length + ' / ' + rows.length + ' reviewed'; try { localStorage.setItem(key, JSON.stringify(rows)); } catch {} }
try { const saved = JSON.parse(localStorage.getItem(key) ?? '[]'); for (const row of saved) { const card = cards.find(c => c.dataset.pair === row.pair_id); if (!card) continue; if (['A','B','tie','neither'].includes(row.preference)) card.querySelector('input[value="' + row.preference + '"]').checked = true; card.querySelector('textarea').value = row.notes ?? ''; } } catch {}
document.addEventListener('input', update); update();
document.querySelector('#export').addEventListener('click', () => { const result = {manifest_sha256: manifest, reviewed_at: new Date().toISOString(), answers: answers()}; const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], {type:'application/json'})); const link = document.createElement('a'); link.href = url; link.download = 'room-review.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
</script></body></html>""".replace("DIGEST", digest)
    (output / "index.html").write_text(page)
