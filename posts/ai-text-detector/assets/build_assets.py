"""Render measured figures for the AI-text detector guide from the headline result."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULT = json.loads((ROOT / "results" / "summary.json").read_text())

BG = "#f7f4ef"
CARD = "#ffffff"
INK = "#1a1a1a"
MUTED = "#5c5c5c"
LINE = "#ddd4c6"
ACCENT = "#c45c26"
STEEL = "#3d6b8a"
GREEN = "#3d6b4f"
GRAY = "#7a8b99"
TAN = "#c4a484"
CREAM = "#f4dccb"

ID = RESULT["distilbert_id"]["at_0_5"]
MED = RESULT["distilbert_medicine"]["at_0_5"]
RAID = RESULT["distilbert_raid"]["at_0_5"]
TFIDF_ID = RESULT["tfidf_id"]["at_0_5"]
TFIDF_MED = RESULT["tfidf_medicine"]["at_0_5"]
TFIDF_RAID = RESULT["tfidf_raid"]["at_0_5"]
DATA = RESULT["data"]
HISTORY = RESULT["history"]


def write(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def hbar_chart(
    *,
    title: str,
    subtitle: str,
    desc: str,
    rows: list[tuple[str, str, float, str]],
    footnote: str,
    value_fmt=None,
    height: int = 520,
    max_value: float | None = None,
) -> str:
    width, pad_l, pad_r, pad_t = 1200, 280, 80, 118
    inner_w = width - pad_l - pad_r
    ceiling = max_value if max_value is not None else max(row[2] for row in rows)
    ceiling = max(ceiling, 1e-6)
    bar_h = 52
    gap = 28
    fmt = value_fmt or (lambda v: pct(v))
    blocks = []
    for i, (label, note, value, color) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        w = max(12.0, inner_w * (value / ceiling))
        value_x = pad_l + w + 14
        anchor = "start"
        fill_value = INK
        if w > 220:
            value_x = pad_l + w - 14
            anchor = "end"
            fill_value = "#fff"
        blocks.append(
            f"""
  <text x="{pad_l - 16}" y="{y + 22}" font-size="16" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">{label}</text>
  <text x="{pad_l - 16}" y="{y + 42}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{note}</text>
  <rect x="{pad_l}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>
  <text x="{value_x:.1f}" y="{y + 34}" font-size="18" font-weight="700" fill="{fill_value}" text-anchor="{anchor}" font-family="system-ui, sans-serif">{fmt(value)}</text>
"""
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">{title}</title>
  <desc id="d">{desc}</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">{title}</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">{subtitle}</text>
  {"".join(blocks)}
  <text x="48" y="{height - 22}" font-size="13" fill="{MUTED}" font-family="system-ui, sans-serif">{footnote}</text>
</svg>
"""


def recall_story() -> str:
    caught_id = ID["ai_recall"]
    caught_raid = RAID["ai_recall"]
    width, height, pad = 1200, 420, 48
    bar_w = width - 2 * pad
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">DistilBERT flags almost every ChatGPT answer and most GPT-4 and Llama answers walk through</title>
  <desc id="d">AI recall is 99.9 percent on HC3 ChatGPT answers and 22.8 percent on RAID GPT-4 and Llama-chat text.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Change the generator</text>
  <text x="{pad}" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Same DistilBERT weights. Same 0.5 threshold. AI recall only.</text>
  <text x="{pad}" y="128" font-size="15" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">HC3 holdout · ChatGPT</text>
  <rect x="{pad}" y="140" width="{bar_w * caught_id:.1f}" height="64" fill="{STEEL}"/>
  <rect x="{pad + bar_w * caught_id:.1f}" y="140" width="{max(bar_w * (1 - caught_id), 2):.1f}" height="64" fill="{LINE}"/>
  <text x="{pad + 20}" y="182" font-size="22" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">flagged {pct(caught_id)}</text>
  <text x="{pad}" y="248" font-size="15" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">RAID · GPT-4 + Llama-chat</text>
  <rect x="{pad}" y="260" width="{bar_w * caught_raid:.1f}" height="64" fill="{ACCENT}"/>
  <rect x="{pad + bar_w * caught_raid:.1f}" y="260" width="{bar_w * (1 - caught_raid):.1f}" height="64" fill="{CREAM}"/>
  <text x="{pad + 16}" y="302" font-size="18" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">flagged {pct(caught_raid)}</text>
  <text x="{pad + bar_w - 16}" y="302" font-size="18" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">walked through {pct(1 - caught_raid)}</text>
  <text x="{pad}" y="388" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">2,251 ChatGPT answers on HC3 · 400 RAID generations (200 GPT-4, 200 Llama-chat) · no adversarial attacks</text>
</svg>
"""


def accuracy_bars() -> str:
    return hbar_chart(
        title="Accuracy across three tests",
        subtitle="DistilBERT in steel, TF-IDF in gray. Chance on the RAID mix is 50%.",
        desc="DistilBERT accuracy is 99.4 percent on HC3, 98.1 percent on medicine, and 48.3 percent on RAID.",
        rows=[
            ("HC3 · DistilBERT", "4,502 held-out answers", ID["accuracy"], STEEL),
            ("HC3 · TF-IDF", "same split", TFIDF_ID["accuracy"], GRAY),
            ("Medicine · DistilBERT", "still ChatGPT, new domain", MED["accuracy"], STEEL),
            ("Medicine · TF-IDF", "1,244 questions held out", TFIDF_MED["accuracy"], GRAY),
            ("RAID · DistilBERT", "GPT-4 + Llama + human", RAID["accuracy"], ACCENT),
            ("RAID · TF-IDF", "600 streamed rows", TFIDF_RAID["accuracy"], TAN),
        ],
        footnote="Threshold 0.5. RAID is 200 human, 200 GPT-4, and 200 Llama-chat documents.",
        height=620,
        max_value=1.0,
    )


def fpr_bars() -> str:
    return hbar_chart(
        title="Human false-positive rate",
        subtitle="How often a person gets called a machine. This is the number a classroom or newsroom would feel.",
        desc="DistilBERT flags 1.0 percent of HC3 humans, 3.5 percent of medicine humans, and 0.5 percent of RAID humans.",
        rows=[
            ("HC3 · DistilBERT", "23 of 2,251 humans", ID["human_fpr"], ACCENT),
            ("HC3 · TF-IDF", "55 of 2,251 humans", TFIDF_ID["human_fpr"], TAN),
            ("Medicine · DistilBERT", "43 of 1,244 humans", MED["human_fpr"], ACCENT),
            ("Medicine · TF-IDF", "74 of 1,244 humans", TFIDF_MED["human_fpr"], TAN),
            ("RAID · DistilBERT", "1 of 200 humans", RAID["human_fpr"], ACCENT),
            ("RAID · TF-IDF", "16 of 200 humans", TFIDF_RAID["human_fpr"], TAN),
        ],
        footnote="Same 0.5 threshold as the accuracy chart. DistilBERT is conservative on RAID and still misses most of the new generators.",
        height=620,
        max_value=0.20,
    )


def dataset_split() -> str:
    train_q = DATA["train_questions"]
    eval_q = DATA["eval_questions"]
    med_q = DATA["domain_questions"]
    total = train_q + eval_q + med_q
    width, height, pad = 1200, 300, 48
    bar_w = width - 2 * pad
    w_train = bar_w * train_q / total
    w_eval = bar_w * eval_q / total
    w_med = bar_w * med_q / total
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">HC3 questions are split by question, with medicine held out</title>
  <desc id="d">20,268 training questions, 2,251 in-distribution holdout questions, and 1,244 medicine questions.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="48" font-size="26" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Split by question, not by answer</text>
  <rect x="{pad}" y="118" width="{w_train:.1f}" height="72" fill="{STEEL}"/>
  <rect x="{pad + w_train:.1f}" y="118" width="{w_eval:.1f}" height="72" fill="{GREEN}"/>
  <rect x="{pad + w_train + w_eval:.1f}" y="118" width="{w_med:.1f}" height="72" fill="{ACCENT}"/>
  <text x="{pad + 20}" y="164" font-size="20" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">train {train_q:,}</text>
  <rect x="{pad}" y="210" width="14" height="14" rx="3" fill="{STEEL}"/>
  <text x="{pad + 22}" y="222" font-size="15" fill="{INK}" font-family="system-ui, sans-serif">train {train_q:,}</text>
  <rect x="{pad + 220}" y="210" width="14" height="14" rx="3" fill="{GREEN}"/>
  <text x="{pad + 242}" y="222" font-size="15" fill="{INK}" font-family="system-ui, sans-serif">HC3 holdout {eval_q:,}</text>
  <rect x="{pad + 500}" y="210" width="14" height="14" rx="3" fill="{ACCENT}"/>
  <text x="{pad + 522}" y="222" font-size="15" fill="{INK}" font-family="system-ui, sans-serif">medicine {med_q:,}</text>
  <text x="{pad}" y="268" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">{total:,} English HC3 questions after dropping short answers. Each question contributes one human answer and one ChatGPT answer.</text>
</svg>
"""


def detector_flow() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="420" viewBox="0 0 1200 420" role="img" aria-labelledby="t d">
  <title id="t">An answer goes into DistilBERT, which predicts human or AI</title>
  <desc id="d">The classifier reads the answer text only. On HC3 it almost always flags ChatGPT. On RAID most GPT-4 and Llama answers are called human.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">What the detector does</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">It does not answer the question. It only reads the answer.</text>
  <rect x="48" y="130" width="240" height="160" rx="16" fill="{CARD}" stroke="{LINE}" stroke-width="2"/>
  <text x="68" y="188" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Answer text</text>
  <text x="68" y="216" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">HC3 or RAID</text>
  <text x="68" y="240" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">human or model</text>
  <path d="M298 210 H338" stroke="{GRAY}" stroke-width="3" fill="none" marker-end="url(#a)"/>
  <rect x="348" y="118" width="300" height="184" rx="16" fill="{CREAM}" stroke="{ACCENT}" stroke-width="3"/>
  <text x="368" y="168" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">DistilBERT</text>
  <text x="368" y="196" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">67M params · 2-class head</text>
  <text x="368" y="220" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">threshold 0.5</text>
  <text x="368" y="256" font-size="14" font-weight="700" fill="{ACCENT}" font-family="system-ui, sans-serif">P(this is AI)</text>
  <path d="M648 160 H698" stroke="{GREEN}" stroke-width="3" fill="none" marker-end="url(#ag)"/>
  <path d="M648 260 H698" stroke="{STEEL}" stroke-width="3" fill="none" marker-end="url(#as)"/>
  <rect x="708" y="96" width="444" height="112" rx="16" fill="#d5e8dd" stroke="{GREEN}" stroke-width="3"/>
  <text x="728" y="140" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Called human</text>
  <text x="728" y="168" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">HC3 humans: {pct(1 - ID["human_fpr"])} correct</text>
  <text x="728" y="190" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">RAID AI: {pct(1 - RAID["ai_recall"])} sneak through</text>
  <rect x="708" y="232" width="444" height="112" rx="16" fill="{CARD}" stroke="{STEEL}" stroke-width="3"/>
  <text x="728" y="276" font-size="18" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Called AI</text>
  <text x="728" y="304" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">HC3 ChatGPT: {pct(ID["ai_recall"])} caught</text>
  <text x="728" y="326" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">RAID humans flagged: {pct(RAID["human_fpr"])}</text>
  <text x="48" y="390" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">Measured on run_ea7b0d5c56f928808c3096a232e3e0bd. TF-IDF is the laptop baseline, not shown here.</text>
  <defs>
    <marker id="a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GRAY}"/></marker>
    <marker id="as" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{STEEL}"/></marker>
    <marker id="ag" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{GREEN}"/></marker>
  </defs>
</svg>
"""


def training_curves() -> str:
    width, height, pad = 1200, 520, 80
    losses = [row[k] for row in HISTORY for k in ("train_loss", "eval_loss")]
    accs = [row["eval_accuracy"] for row in HISTORY]
    l_lo, l_hi = 0.0, max(losses) * 1.25
    a_lo, a_hi = 0.988, 0.996

    def x_at(epoch: int) -> float:
        return pad + (width - 2 * pad) * ((epoch - 1) / max(len(HISTORY) - 1, 1))

    def y_loss(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - l_lo) / (l_hi - l_lo))

    def y_acc(value: float) -> float:
        return pad + (height - 2 * pad) * (1 - (value - a_lo) / (a_hi - a_lo))

    def poly(key: str, yfn) -> str:
        return " ".join(f"{x_at(row['epoch']):.1f},{yfn(row[key]):.1f}" for row in HISTORY)

    e1, e2 = HISTORY
    dots = []
    for row in HISTORY:
        x = x_at(row["epoch"])
        dots.append(f'<circle cx="{x:.1f}" cy="{y_loss(row["train_loss"]):.1f}" r="7" fill="{STEEL}"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_loss(row["eval_loss"]):.1f}" r="7" fill="{ACCENT}"/>')
        dots.append(f'<circle cx="{x:.1f}" cy="{y_acc(row["eval_accuracy"]):.1f}" r="7" fill="{GREEN}"/>')
    grid = []
    for tick in (0.00, 0.02, 0.04):
        y = y_loss(tick)
        grid.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" stroke="{LINE}"/>'
            f'<text x="{pad - 12}" y="{y + 5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">{tick:.2f}</text>'
        )
    for tick in (0.990, 0.992, 0.994, 0.996):
        y = y_acc(tick)
        grid.append(
            f'<text x="{width - pad + 12}" y="{y + 5:.1f}" font-size="13" fill="{GREEN}" text-anchor="start" font-family="system-ui, sans-serif">{tick * 100:.1f}%</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">DistilBERT train loss, eval loss, and held-out accuracy over two epochs</title>
  <desc id="d">Train loss fell from 0.044 to 0.009. Eval loss fell from 0.023 to 0.018. Held-out accuracy rose from 99.22 percent to 99.42 percent.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="44" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Two epochs on 40,536 answers</text>
  <text x="48" y="74" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Loss on the left axis. Held-out accuracy is the dashed green line, right axis 99.0%–99.6%.</text>
  {"".join(grid)}
  <polyline fill="none" stroke="{STEEL}" stroke-width="3.5" points="{poly("train_loss", y_loss)}"/>
  <polyline fill="none" stroke="{ACCENT}" stroke-width="3.5" points="{poly("eval_loss", y_loss)}"/>
  <polyline fill="none" stroke="{GREEN}" stroke-width="3.5" stroke-dasharray="8 6" points="{poly("eval_accuracy", y_acc)}"/>
  {"".join(dots)}
  <text x="{x_at(1):.1f}" y="{height - pad + 28}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">epoch 1</text>
  <text x="{x_at(2):.1f}" y="{height - pad + 28}" font-size="14" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">epoch 2</text>
  <rect x="48" y="{height - 28}" width="12" height="12" fill="{STEEL}"/>
  <text x="66" y="{height - 17}" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">train loss {e1["train_loss"]:.3f} → {e2["train_loss"]:.3f}</text>
  <rect x="360" y="{height - 28}" width="12" height="12" fill="{ACCENT}"/>
  <text x="378" y="{height - 17}" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">eval loss {e1["eval_loss"]:.3f} → {e2["eval_loss"]:.3f}</text>
  <rect x="680" y="{height - 28}" width="12" height="12" fill="{GREEN}"/>
  <text x="698" y="{height - 17}" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">holdout acc {pct(e1["eval_accuracy"])} → {pct(e2["eval_accuracy"])}</text>
</svg>
"""


def _confusion_counts(split: dict) -> tuple[int, int, int, int]:
    human_n = split["human_n"]
    ai_n = split["ai_n"]
    fp = int(round(human_n * split["human_fpr"]))
    tn = human_n - fp
    tp = int(round(ai_n * split["ai_recall"]))
    fn = ai_n - tp
    return tn, fp, fn, tp


def _matrix_cell(
    x: float, y: float, w: float, h: float, value: int, ceiling: int, label: str, *, error: bool
) -> str:
    t = 0 if ceiling <= 0 else max(value / ceiling, 0.12)
    if error:
        fill = f"rgb({int(244 - 48 * t)},{int(220 - 128 * t)},{int(203 - 165 * t)})"
        ink = "#fff" if t > 0.45 else INK
    else:
        fill = f"rgb({int(180 - 119 * t)},{int(200 - 93 * t)},{int(210 - 72 * t)})"
        ink = "#fff" if t > 0.40 else INK
    return f"""
  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10" fill="{fill}"/>
  <text x="{x + w / 2:.1f}" y="{y + h / 2 - 6:.1f}" font-size="28" font-weight="700" fill="{ink}" text-anchor="middle" font-family="system-ui, sans-serif">{value:,}</text>
  <text x="{x + w / 2:.1f}" y="{y + h / 2 + 20:.1f}" font-size="13" fill="{ink}" text-anchor="middle" font-family="system-ui, sans-serif">{label}</text>
"""


def confusion_matrices() -> str:
    id_tn, id_fp, id_fn, id_tp = _confusion_counts(ID)
    raid_tn, raid_fp, raid_fn, raid_tp = _confusion_counts(RAID)
    width, height = 1200, 560
    cell_w, cell_h, gap = 196, 118, 12
    left_x, right_x = 86, 686
    top = 186

    def panel(ox: float, title: str, subtitle: str, tn: int, fp: int, fn: int, tp: int) -> str:
        ceiling = max(tn, fp, fn, tp)
        return f"""
  <text x="{ox}" y="132" font-size="20" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">{title}</text>
  <text x="{ox}" y="154" font-size="13" fill="{MUTED}" font-family="system-ui, sans-serif">{subtitle}</text>
  <text x="{ox + 48 + cell_w / 2:.1f}" y="{top - 8}" font-size="13" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">called human</text>
  <text x="{ox + 48 + cell_w + gap + cell_w / 2:.1f}" y="{top - 8}" font-size="13" fill="{MUTED}" text-anchor="middle" font-family="system-ui, sans-serif">called AI</text>
  <text x="{ox + 36}" y="{top + cell_h / 2 + 5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">is human</text>
  <text x="{ox + 36}" y="{top + cell_h + gap + cell_h / 2 + 5:.1f}" font-size="13" fill="{MUTED}" text-anchor="end" font-family="system-ui, sans-serif">is AI</text>
  {_matrix_cell(ox + 48, top, cell_w, cell_h, tn, ceiling, "true negative", error=False)}
  {_matrix_cell(ox + 48 + cell_w + gap, top, cell_w, cell_h, fp, ceiling, "false positive", error=True)}
  {_matrix_cell(ox + 48, top + cell_h + gap, cell_w, cell_h, fn, ceiling, "missed AI", error=True)}
  {_matrix_cell(ox + 48 + cell_w + gap, top + cell_h + gap, cell_w, cell_h, tp, ceiling, "caught AI", error=False)}
"""

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">
  <title id="t">Confusion matrices on HC3 and RAID at threshold 0.5</title>
  <desc id="d">On HC3, DistilBERT is almost a perfect diagonal. On RAID, 309 of 400 GPT-4 and Llama answers are called human.</desc>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="48" y="48" font-size="28" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">Same threshold, two tests</text>
  <text x="48" y="80" font-size="16" fill="{MUTED}" font-family="system-ui, sans-serif">Counts at 0.5. Steel is correct. Orange is wrong. The RAID miss is the big orange square.</text>
  {panel(left_x, "HC3 holdout", "2,251 human + 2,251 ChatGPT", id_tn, id_fp, id_fn, id_tp)}
  {panel(right_x, "RAID", "200 human + 200 GPT-4 + 200 Llama", raid_tn, raid_fp, raid_fn, raid_tp)}
  <text x="48" y="536" font-size="14" fill="{MUTED}" font-family="system-ui, sans-serif">HC3 false positives: 23 humans. RAID false positives: 1 human. RAID missed AI: 309 of 400.</text>
</svg>
"""


def social_card() -> str:
    width, height = 1200, 630
    pad = 72
    bar_w = width - 2 * pad
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="t">
  <title id="t">A $0.38 AI-text detector</title>
  <rect width="100%" height="100%" fill="{BG}"/>
  <text x="{pad}" y="88" font-size="20" fill="{MUTED}" font-family="system-ui, sans-serif">Let’s use compute · post #6</text>
  <text x="{pad}" y="168" font-size="52" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">A $0.38 AI-text detector</text>
  <text x="{pad}" y="220" font-size="22" fill="{MUTED}" font-family="system-ui, sans-serif">99.4% on ChatGPT. 48% when the generator changes.</text>
  <text x="{pad}" y="300" font-size="16" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">ChatGPT answers flagged</text>
  <rect x="{pad}" y="312" width="{bar_w * ID["ai_recall"]:.1f}" height="48" fill="{STEEL}"/>
  <text x="{pad + 16}" y="344" font-size="18" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">{pct(ID["ai_recall"])}</text>
  <text x="{pad}" y="410" font-size="16" font-weight="700" fill="{INK}" font-family="system-ui, sans-serif">GPT-4 + Llama answers flagged</text>
  <rect x="{pad}" y="422" width="{bar_w * RAID["ai_recall"]:.1f}" height="48" fill="{ACCENT}"/>
  <rect x="{pad + bar_w * RAID["ai_recall"]:.1f}" y="422" width="{bar_w * (1 - RAID["ai_recall"]):.1f}" height="48" fill="{CREAM}"/>
  <text x="{pad + 12}" y="454" font-size="16" font-weight="700" fill="#fff" font-family="system-ui, sans-serif">{pct(RAID["ai_recall"])}</text>
  <text x="{width - pad}" y="454" font-size="16" font-weight="700" fill="{INK}" text-anchor="end" font-family="system-ui, sans-serif">{pct(1 - RAID["ai_recall"])} missed</text>
  <text x="{pad}" y="560" font-size="18" fill="{ACCENT}" font-family="system-ui, sans-serif">letsusecompute.com/posts/ai-text-detector</text>
</svg>
"""


def main() -> None:
    write(ROOT / "recall_story.svg", recall_story())
    write(ROOT / "accuracy_bars.svg", accuracy_bars())
    write(ROOT / "fpr_bars.svg", fpr_bars())
    write(ROOT / "dataset_split.svg", dataset_split())
    write(ROOT / "detector_flow.svg", detector_flow())
    write(ROOT / "training_curves.svg", training_curves())
    write(ROOT / "confusion_matrices.svg", confusion_matrices())
    write(ROOT / "social_card.svg", social_card())
    # Keep the old filenames as aliases so existing links do not 404.
    write(ROOT / "ood_story.svg", recall_story())
    write(ROOT / "metrics.svg", accuracy_bars())


if __name__ == "__main__":
    main()
