"""
Crypto Sentiment PDF Report Generator
--------------------------------------
Usage:
    python generate_report.py
    python generate_report.py --input <path_to_your_file.jsonl> --output <report.pdf>

Reads a JSONL file of scored crypto news articles and produces a one-page
PDF sentiment summary with charts, source rankings, headlines table, and
a market analysis narrative.
"""

import argparse
import json
import io
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.utils import ImageReader

# ─── Colour palette ──────────────────────────────────────────────────────────
C_BG        = "#0D1117"   # page background
C_CARD      = "#161B22"   # card/panel background
C_BORDER    = "#30363D"   # subtle borders
C_TEXT      = "#E6EDF3"   # primary text
C_MUTED     = "#8B949E"   # secondary text
C_BULL      = "#3FB950"   # green  – bullish
C_BEAR      = "#F85149"   # red    – bearish
C_NEUTRAL   = "#D29922"   # amber  – neutral
C_ACCENT    = "#58A6FF"   # blue   – highlights / titles
C_ACCENT2   = "#BC8CFF"   # purple – secondary accent

MAX_SOURCES = 6   # max sources to show in bar chart
MAX_HEADLINES = 5  # max headlines in the table

# ─── Helpers ─────────────────────────────────────────────────────────────────

def hex_to_rgb01(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def rl_color(h):
    r, g, b = hex_to_rgb01(h)
    return colors.Color(r, g, b)

def score_to_color(score):
    if score >= 0.15:
        return C_BULL
    if score <= -0.15:
        return C_BEAR
    return C_NEUTRAL

def label_signal(score):
    if score >= 0.5:  return ("STRONGLY BULLISH", C_BULL)
    if score >= 0.15: return ("BULLISH",          C_BULL)
    if score > -0.15: return ("NEUTRAL",           C_NEUTRAL)
    if score > -0.5:  return ("BEARISH",           C_BEAR)
    return              ("STRONGLY BEARISH",        C_BEAR)

def truncate(text, n):
    return text if len(text) <= n else text[:n - 1] + "…"

# ─── Data processing ─────────────────────────────────────────────────────────

def load_data(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def analyse(records):
    overall_score = sum(r["score"] for r in records) / len(records)

    # Sentiment distribution
    dist = defaultdict(int)
    for r in records:
        lbl = r.get("label", "neutral").lower()
        if lbl not in ("bullish", "bearish"):
            lbl = "neutral"
        dist[lbl] += 1

    # Per-source stats
    src_data = defaultdict(list)
    for r in records:
        src_data[r["source"]].append(r["score"])
    source_stats = {
        s: {"avg": sum(v) / len(v), "count": len(v)}
        for s, v in src_data.items()
    }

    # Per-crypto stats — one article may mention multiple cryptos
    crypto_data = defaultdict(list)
    for r in records:
        for c in r.get("cryptos", []):
            crypto_data[c].append(r["score"])
    crypto_stats = {
        c: {"avg": sum(v) / len(v), "count": len(v)}
        for c, v in crypto_data.items()
    }

    # Top headlines: pick diverse extremes (most bullish + most bearish)
    sorted_bull = sorted([r for r in records if r["score"] > 0],
                         key=lambda x: x["score"], reverse=True)
    sorted_bear = sorted([r for r in records if r["score"] < 0],
                         key=lambda x: x["score"])
    top = []
    seen = set()
    for r in sorted_bull + sorted_bear:
        if r["url"] not in seen:
            top.append(r)
            seen.add(r["url"])
        if len(top) == MAX_HEADLINES:
            break

    # Date range
    dates = [r["published_at"] for r in records if r.get("published_at")]
    date_min = min(dates) if dates else "N/A"
    date_max = max(dates) if dates else "N/A"

    return {
        "overall": overall_score,
        "dist": dict(dist),
        "sources": source_stats,
        "cryptos": crypto_stats,
        "headlines": top,
        "total": len(records),
        "date_min": date_min,
        "date_max": date_max,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

# ─── Chart generators (return PNG bytes) ─────────────────────────────────────

def make_gauge(score, w=280, h=200):
    """Half-donut gauge showing overall score."""
    fig, ax = plt.subplots(figsize=(w/100, h/100), facecolor=C_BG)
    ax.set_aspect("equal")

    # Draw arc segments
    segments = [
        (-1.0, -0.5, C_BEAR),
        (-0.5, -0.15, "#8B4040"),
        (-0.15, 0.15, "#6B6B2A"),
        (0.15, 0.5,  "#2A6B40"),
        (0.5,  1.0,  C_BULL),
    ]
    r_out, r_in = 1.0, 0.55
    for lo, hi, col in segments:
        ang_lo = 180 - (lo + 1) / 2 * 180
        ang_hi = 180 - (hi + 1) / 2 * 180
        theta = np.linspace(np.radians(ang_hi), np.radians(ang_lo), 60)
        x_out = r_out * np.cos(theta)
        y_out = r_out * np.sin(theta)
        x_in  = r_in  * np.cos(theta[::-1])
        y_in  = r_in  * np.sin(theta[::-1])
        ax.fill(np.concatenate([x_out, x_in]),
                np.concatenate([y_out, y_in]),
                color=col, lw=0)

    # Needle
    needle_ang = np.radians(180 - (score + 1) / 2 * 180)
    ax.plot([0, 0.72 * np.cos(needle_ang)],
            [0, 0.72 * np.sin(needle_ang)],
            color="white", lw=2.5, zorder=5)
    ax.add_patch(plt.Circle((0, 0), 0.06, color="white", zorder=6))

    # Score text
    ax.text(0, -0.18, f"{score:+.2f}", ha="center", va="center",
            fontsize=18, fontweight="bold",
            color=score_to_color(score), fontfamily="monospace")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.35, 1.1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=C_BG, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def make_donut(dist, w=220, h=220):
    """Sentiment distribution donut chart."""
    labels = []
    sizes  = []
    clrs   = []
    mapping = {"bullish": C_BULL, "bearish": C_BEAR, "neutral": C_NEUTRAL}
    for lbl, col in mapping.items():
        cnt = dist.get(lbl, 0)
        if cnt:
            labels.append(f"{lbl.capitalize()}\n{cnt}")
            sizes.append(cnt)
            clrs.append(col)

    fig, ax = plt.subplots(figsize=(w/100, h/100), facecolor=C_BG)
    wedges, _ = ax.pie(
        sizes, colors=clrs, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=C_BG, linewidth=2),
        counterclock=False,
    )
    # Custom legend
    patches = [mpatches.Patch(color=c, label=l) for c, l in
               zip(clrs, [lb.split("\n")[0] + " " + lb.split("\n")[1] for lb in labels])]
    ax.legend(handles=patches, loc="center", fontsize=7.5,
              frameon=False,
              labelcolor=C_TEXT,
              handlelength=1.2, handleheight=1.0)
    ax.set_facecolor(C_BG)
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=C_BG, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def make_source_bars(source_stats, w=480, h=200):
    """Horizontal bar chart of average sentiment per source."""
    # pick top MAX_SOURCES by article count, then sort by avg score
    top = sorted(source_stats.items(), key=lambda x: -x[1]["count"])[:MAX_SOURCES]
    top = sorted(top, key=lambda x: x[1]["avg"])

    names  = [t[0] for t in top]
    scores = [t[1]["avg"] for t in top]
    counts = [t[1]["count"] for t in top]
    bar_colors = [score_to_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(w/100, h/100), facecolor=C_BG)
    ax.set_facecolor(C_CARD)

    bars = ax.barh(names, scores, color=bar_colors,
                   height=0.55, edgecolor="none")

    # Zero line
    ax.axvline(0, color=C_BORDER, linewidth=1)

    # Value + count labels
    for bar, score, cnt in zip(bars, scores, counts):
        xpos = bar.get_width()
        ha = "left" if xpos >= 0 else "right"
        offset = 0.03 if xpos >= 0 else -0.03
        ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                f"{score:+.2f}  ({cnt})",
                va="center", ha=ha, fontsize=8,
                color=C_TEXT, fontfamily="monospace")

    ax.set_xlim(-1.15, 1.15)
    ax.set_xlabel("Average Sentiment Score", color=C_MUTED, fontsize=8)
    ax.tick_params(colors=C_TEXT, labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_color(C_MUTED)
    ax.tick_params(axis="x", colors=C_MUTED)
    ax.tick_params(axis="y", colors=C_TEXT)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=3)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1.0", "-0.5", "0", "+0.5", "+1.0"],
                        fontsize=7, color=C_MUTED)

    # Dynamically compute left margin so long source names are never clipped
    max_chars = max(len(n) for n in names)
    left_margin = min(0.45, max(0.15, max_chars * 0.018 + 0.05))

    fig.subplots_adjust(left=left_margin, right=0.88, top=0.92, bottom=0.22)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=C_BG, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def make_crypto_bars(crypto_stats, w=320, h=160):
    """Horizontal bar chart for crypto breakdown."""
    top = sorted(crypto_stats.items(), key=lambda x: -x[1]["count"])[:6]
    top = sorted(top, key=lambda x: x[1]["avg"])

    names  = [t[0] for t in top]
    scores = [t[1]["avg"] for t in top]
    bar_colors = [score_to_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(w/100, h/100), facecolor=C_BG)
    ax.set_facecolor(C_CARD)
    ax.barh(names, scores, color=bar_colors, height=0.5, edgecolor="none")
    ax.axvline(0, color=C_BORDER, linewidth=1)
    for i, (score, name) in enumerate(zip(scores, names)):
        xpos = score
        ha = "left" if xpos >= 0 else "right"
        offset = 0.03 if xpos >= 0 else -0.03
        ax.text(xpos + offset, i, f"{score:+.2f}",
                va="center", ha=ha, fontsize=7.5,
                color=C_TEXT, fontfamily="monospace")
    ax.set_xlim(-1.15, 1.15)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", colors=C_MUTED)
    ax.tick_params(axis="y", colors=C_TEXT)
    ax.yaxis.set_tick_params(length=0)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["-1", "-.5", "0", "+.5", "+1"],
                        fontsize=6.5, color=C_MUTED)
    # Dynamically compute left margin so long crypto names are never clipped
    max_chars = max(len(n) for n in names)
    left_margin = min(0.50, max(0.18, max_chars * 0.024 + 0.05))

    fig.subplots_adjust(left=left_margin, right=0.88, top=0.95, bottom=0.18)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=C_BG, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ─── Narrative generator ─────────────────────────────────────────────────────

def build_narrative(data):
    overall = data["overall"]
    dist    = data["dist"]
    sources = data["sources"]
    cryptos = data["cryptos"]
    total   = data["total"]

    signal, _ = label_signal(overall)
    bull_pct = dist.get("bullish", 0) / total * 100
    bear_pct = dist.get("bearish", 0) / total * 100
    neu_pct  = dist.get("neutral",  0) / total * 100

    # Most bullish / bearish source
    if sources:
        best_src = max(sources, key=lambda s: sources[s]["avg"])
        worst_src = min(sources, key=lambda s: sources[s]["avg"])
    else:
        best_src = worst_src = "N/A"

    # Most covered crypto
    if cryptos:
        top_crypto = max(cryptos, key=lambda c: cryptos[c]["count"])
        top_crypto_score = cryptos[top_crypto]["avg"]
        top_crypto_signal = "bullish" if top_crypto_score > 0.15 else \
                            "bearish" if top_crypto_score < -0.15 else "neutral"
    else:
        top_crypto = "Unknown"
        top_crypto_signal = "neutral"

    lines = [
        f"Across {total} article{'s' if total != 1 else ''} analysed, the aggregate market "
        f"sentiment reads <b>{signal}</b> with a composite score of <b>{overall:+.2f}</b>. "
        f"Bullish coverage accounts for {bull_pct:.0f}% of articles, bearish for "
        f"{bear_pct:.0f}%, and neutral for {neu_pct:.0f}%.",

        f"Among tracked sources, <b>{best_src}</b> showed the most optimistic tone "
        f"(avg {sources[best_src]['avg']:+.2f}), while <b>{worst_src}</b> carried "
        f"the most cautious outlook "
        f"(avg {sources[worst_src]['avg']:+.2f})." if best_src != "N/A" else "",

        f"<b>{top_crypto}</b> attracted the most coverage and is broadly viewed as "
        f"<b>{top_crypto_signal}</b> based on current reporting." if top_crypto != "Unknown" else "",

        "Investors should weigh these signals alongside on-chain data and macro "
        "conditions. This report is auto-generated for informational purposes only "
        "and does not constitute financial advice.",
    ]
    return "  ".join(l for l in lines if l)

# ─── PDF layout ──────────────────────────────────────────────────────────────

PAGE_W, PAGE_H = A4          # 595 x 842 pt
MARGIN = 14 * mm

def draw_rounded_rect(c, x, y, w, h, r=4, fill=None, stroke=None):
    c.saveState()
    if fill:
        c.setFillColor(rl_color(fill))
    if stroke:
        c.setStrokeColor(rl_color(stroke))
        c.setLineWidth(0.5)
    path = c.beginPath()
    path.roundRect(x, y, w, h, r)
    c.drawPath(path, fill=1 if fill else 0, stroke=1 if stroke else 0)
    c.restoreState()

def put_image_bytes(c, img_bytes, x, y, w, h):
    buf = io.BytesIO(img_bytes)
    img = ImageReader(buf)
    c.drawImage(img, x, y, width=w, height=h, preserveAspectRatio=True, mask="auto")

def draw_text(c, text, x, y, size=9, color=C_TEXT, bold=False, align="left"):
    c.saveState()
    c.setFillColor(rl_color(color))
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)
    c.restoreState()

def draw_paragraph(c, html_text, x, y, w, h, size=8, color=C_TEXT, leading=12):
    style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=size,
        leading=leading,
        textColor=rl_color(color),
        wordWrap="CJK",
    )
    p = Paragraph(html_text, style)
    pw, ph = p.wrap(w, h)
    p.drawOn(c, x, y + h - ph)

def build_pdf(data, charts, output_path):
    c = rl_canvas.Canvas(output_path, pagesize=A4)
    W, H = PAGE_W, PAGE_H

    # ── Background ────────────────────────────────────────────────────────────
    c.setFillColor(rl_color(C_BG))
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Header bar ───────────────────────────────────────────────────────────
    HDR_H = 38 * mm
    draw_rounded_rect(c, MARGIN, H - MARGIN - HDR_H,
                      W - 2*MARGIN, HDR_H, r=5, fill=C_CARD, stroke=C_BORDER)

    draw_text(c, "CRYPTO SENTIMENT REPORT",
              MARGIN + 6*mm, H - MARGIN - 14*mm,
              size=17, color=C_ACCENT, bold=True)

    signal_text, signal_col = label_signal(data["overall"])
    draw_text(c, f"Market Signal: {signal_text}",
              MARGIN + 6*mm, H - MARGIN - 22*mm,
              size=10, color=signal_col, bold=True)

    date_range = (f"{data['date_min']}  →  {data['date_max']}"
                  if data['date_min'] != data['date_max'] else data['date_min'])
    draw_text(c, f"Coverage: {date_range}   •   {data['total']} articles   •   Generated {data['generated_at']}",
              MARGIN + 6*mm, H - MARGIN - 30*mm,
              size=8, color=C_MUTED)

    # ── Section: Gauge + Donut ────────────────────────────────────────────────
    SEC1_TOP = H - MARGIN - HDR_H - 4*mm
    SEC1_H   = 52 * mm
    GAUGE_W  = 72 * mm
    DONUT_W  = W - 2*MARGIN - GAUGE_W - 3*mm

    # Gauge card
    draw_rounded_rect(c, MARGIN, SEC1_TOP - SEC1_H, GAUGE_W, SEC1_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "OVERALL SCORE", MARGIN + GAUGE_W/2,
              SEC1_TOP - 7*mm, size=7.5, color=C_MUTED, bold=True, align="center")
    put_image_bytes(c, charts["gauge"],
                    MARGIN + 2*mm, SEC1_TOP - SEC1_H + 2*mm,
                    GAUGE_W - 4*mm, SEC1_H - 10*mm)

    # Donut card
    DONUT_X = MARGIN + GAUGE_W + 3*mm
    draw_rounded_rect(c, DONUT_X, SEC1_TOP - SEC1_H, DONUT_W, SEC1_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "SENTIMENT DISTRIBUTION", DONUT_X + DONUT_W/2,
              SEC1_TOP - 7*mm, size=7.5, color=C_MUTED, bold=True, align="center")
    put_image_bytes(c, charts["donut"],
                    DONUT_X + 2*mm, SEC1_TOP - SEC1_H + 1*mm,
                    DONUT_W - 4*mm, SEC1_H - 10*mm)

    # ── Section: Source bar chart ─────────────────────────────────────────────
    SEC2_TOP = SEC1_TOP - SEC1_H - 4*mm
    SEC2_H   = 55 * mm
    SRC_W    = (W - 2*MARGIN) * 0.6 - 2*mm
    CRYPTO_W = (W - 2*MARGIN) * 0.4 - 1*mm

    draw_rounded_rect(c, MARGIN, SEC2_TOP - SEC2_H, SRC_W, SEC2_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "SOURCE SENTIMENT (avg score  •  article count)",
              MARGIN + SRC_W/2, SEC2_TOP - 7*mm,
              size=7.5, color=C_MUTED, bold=True, align="center")
    put_image_bytes(c, charts["source_bars"],
                    MARGIN + 1*mm, SEC2_TOP - SEC2_H + 1*mm,
                    SRC_W - 2*mm, SEC2_H - 10*mm)

    # Crypto breakdown card
    CRYPTO_X = MARGIN + SRC_W + 3*mm
    draw_rounded_rect(c, CRYPTO_X, SEC2_TOP - SEC2_H, CRYPTO_W, SEC2_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "BY CRYPTOCURRENCY",
              CRYPTO_X + CRYPTO_W/2, SEC2_TOP - 7*mm,
              size=7.5, color=C_MUTED, bold=True, align="center")
    put_image_bytes(c, charts["crypto_bars"],
                    CRYPTO_X + 1*mm, SEC2_TOP - SEC2_H + 1*mm,
                    CRYPTO_W - 2*mm, SEC2_H - 10*mm)

    # ── Section: Headlines table ──────────────────────────────────────────────
    SEC3_TOP = SEC2_TOP - SEC2_H - 4*mm
    ROW_H    = 9.2 * mm
    NCOLS    = len(data["headlines"])
    SEC3_H   = (NCOLS + 1) * ROW_H + 7*mm

    draw_rounded_rect(c, MARGIN, SEC3_TOP - SEC3_H, W - 2*MARGIN, SEC3_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "TOP HEADLINES", MARGIN + (W - 2*MARGIN)/2,
              SEC3_TOP - 5.5*mm, size=7.5, color=C_MUTED, bold=True, align="center")

    # Table header
    COL_TITLE_W = (W - 2*MARGIN) * 0.52
    COL_SRC_W   = (W - 2*MARGIN) * 0.13
    COL_CRYP_W  = (W - 2*MARGIN) * 0.13
    COL_SCOR_W  = (W - 2*MARGIN) * 0.10
    COL_LABL_W  = (W - 2*MARGIN) * 0.12

    HDR_Y = SEC3_TOP - 13*mm
    c.setFillColor(rl_color(C_BORDER))
    c.rect(MARGIN + 1*mm, HDR_Y - 1*mm,
           W - 2*MARGIN - 2*mm, ROW_H * 0.75, fill=1, stroke=0)

    hx = MARGIN + 3*mm
    for label, width in [("HEADLINE", COL_TITLE_W), ("SOURCE", COL_SRC_W),
                          ("ASSET", COL_CRYP_W), ("SCORE", COL_SCOR_W),
                          ("SIGNAL", COL_LABL_W)]:
        draw_text(c, label, hx, HDR_Y, size=7, color=C_MUTED, bold=True)
        hx += width

    for i, rec in enumerate(data["headlines"]):
        row_y = HDR_Y - (i + 1) * ROW_H - 0.5*mm
        # Alternating row background
        if i % 2 == 0:
            c.setFillColor(rl_color("#1A2030"))
            c.rect(MARGIN + 1*mm, row_y - 1.5*mm,
                   W - 2*MARGIN - 2*mm, ROW_H * 0.82, fill=1, stroke=0)

        rx = MARGIN + 3*mm
        title = truncate(rec["title_clean"], 68)
        draw_text(c, title, rx, row_y, size=7.5, color=C_TEXT)
        rx += COL_TITLE_W

        draw_text(c, truncate(rec["source"], 12), rx, row_y, size=7.5, color=C_MUTED)
        rx += COL_SRC_W

        cryptos_list = rec.get("cryptos", [])
        crypto_cell = truncate("/".join(cryptos_list) if cryptos_list else "—", 14)
        draw_text(c, crypto_cell, rx, row_y, size=7.5, color=C_MUTED)
        rx += COL_CRYP_W

        score_col = score_to_color(rec["score"])
        draw_text(c, f"{rec['score']:+.2f}", rx, row_y, size=7.5, color=score_col, bold=True)
        rx += COL_SCOR_W

        lbl_col = C_BULL if "bull" in rec["label"].lower() else \
                  C_BEAR if "bear" in rec["label"].lower() else C_NEUTRAL
        draw_text(c, rec["label"].upper(), rx, row_y, size=7, color=lbl_col, bold=True)

    # ── Section: Market narrative ─────────────────────────────────────────────
    SEC4_TOP = SEC3_TOP - SEC3_H - 4*mm
    SEC4_H   = 34 * mm

    draw_rounded_rect(c, MARGIN, SEC4_TOP - SEC4_H, W - 2*MARGIN, SEC4_H,
                      r=4, fill=C_CARD, stroke=C_BORDER)
    draw_text(c, "MARKET ANALYSIS SUMMARY", MARGIN + 6*mm, SEC4_TOP - 7*mm,
              size=7.5, color=C_MUTED, bold=True)

    narrative = build_narrative(data)
    draw_paragraph(c, narrative,
                   MARGIN + 4*mm, SEC4_TOP - SEC4_H + 2*mm,
                   W - 2*MARGIN - 8*mm, SEC4_H - 12*mm,
                   size=8, color=C_TEXT, leading=11.5)

    # ── Footer ────────────────────────────────────────────────────────────────
    FOOT_Y = MARGIN + 1*mm
    c.setStrokeColor(rl_color(C_BORDER))
    c.setLineWidth(0.4)
    c.line(MARGIN, FOOT_Y + 4*mm, W - MARGIN, FOOT_Y + 4*mm)
    draw_text(c, "Powered by Ollama  •  llama3.2  •  For informational purposes only — not financial advice.",
              W/2, FOOT_Y, size=6.5, color=C_MUTED, align="center")

    c.save()
    print(f"[✓] Report saved → {output_path}")

def export_pdf(input_path, output_path):
    print(f"[→] Loading {input_path}…")
    records = load_data(input_path)
    if not records:
        print("Error: no records found in input file.", file=sys.stderr)
        sys.exit(1)
    print(f"[→] {len(records)} articles loaded.")

    data = analyse(records)
    print(f"[→] Overall score: {data['overall']:+.3f}  |  signal: {label_signal(data['overall'])[0]}")

    print("[→] Rendering charts…")
    charts = {
        "gauge":       make_gauge(data["overall"]),
        "donut":       make_donut(data["dist"]),
        "source_bars": make_source_bars(data["sources"]),
        "crypto_bars": make_crypto_bars(data["cryptos"]),
    }

    print("[→] Building PDF…")
    resolved_output = resolve_output_path(output_path)
    build_pdf(data, charts, str(resolved_output))


def resolve_output_path(output_path: str) -> Path:
    """Always save reports under model/data/report; keep only filename from output_path."""
    script_dir = Path(__file__).resolve().parent
    report_dir = script_dir / "data" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(output_path).name if output_path else "crypto_sentiment_report.pdf"
    return report_dir / filename


def resolve_default_input_path() -> str:
    script_dir = Path(__file__).resolve().parent
    scored_dir = script_dir / "data" / "scored"
    if not scored_dir.exists():
        return ""

    preferred = sorted(
        scored_dir.glob("sentiment_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if preferred:
        return str(preferred[0])

    fallback = sorted(
        scored_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if fallback:
        return str(fallback[0])
    return ""

# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate crypto sentiment PDF report")
    parser.add_argument("--input",  "-i", required=False, help="Path to .jsonl sentiment file")
    parser.add_argument("--output", "-o", default="crypto_sentiment_report.pdf",
                        help="Output PDF filename (always saved to model/data/report)")
    args = parser.parse_args()
    input_path = args.input or resolve_default_input_path()
    if not input_path:
        parser.error(
            "No default scored JSONL found in model/data/scored. "
            "Provide one with --input."
        )
    export_pdf(input_path, args.output)

if __name__ == "__main__":
    main()
