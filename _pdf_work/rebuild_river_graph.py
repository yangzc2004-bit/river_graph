# -*- coding: utf-8 -*-
"""Rebuild river_graph.pdf: drop cover meta block, neutralize teacher wording in §5."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(r"D:\river_graph\river_graph.pdf")
MAP_PATH = Path(r"D:\river_graph\_pdf_work\map.png")

BG = colors.HexColor("#FCFBF8")
INK = colors.HexColor("#1A1F1C")
MUTED = colors.HexColor("#6B7370")
GREEN = colors.HexColor("#1F6B52")
GREEN_DARK = colors.HexColor("#145A44")
GREEN_LIGHT = colors.HexColor("#EAF2EE")
LINE = colors.HexColor("#C9D5CF")
ACCENT = colors.HexColor("#6B5B1E")
ACCENT_BG = colors.HexColor("#F8F5E8")
ACCENT_BAR = colors.HexColor("#8A7318")
TEAL = colors.HexColor("#2F6B6B")

W, H = A4
ML = 18 * mm
MR = 18 * mm
MT = 16 * mm
MB = 16 * mm
CW = W - ML - MR


def register_fonts():
    pdfmetrics.registerFont(TTFont("YH", r"C:\Windows\Fonts\msyh.ttc"))
    pdfmetrics.registerFont(TTFont("YH-Bold", r"C:\Windows\Fonts\msyhbd.ttc"))
    pdfmetrics.registerFont(TTFont("YH-L", r"C:\Windows\Fonts\msyhl.ttc"))
    pdfmetrics.registerFont(TTFont("TimesR", r"C:\Windows\Fonts\times.ttf"))
    pdfmetrics.registerFont(TTFont("TimesB", r"C:\Windows\Fonts\timesbd.ttf"))
    pdfmetrics.registerFont(TTFont("TimesI", r"C:\Windows\Fonts\timesi.ttf"))


register_fonts()
FONT = "YH"
FONT_B = "YH-Bold"
FONT_EN = "TimesR"
FONT_EN_B = "TimesB"
FONT_EN_I = "TimesI"


def st(name, **kw):
    base = dict(fontName=FONT, fontSize=10.5, leading=16, textColor=INK, alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "brand": st("brand", fontName=FONT, fontSize=10, leading=14, textColor=TEAL),
    "cover_h1": st("cover_h1", fontName=FONT_B, fontSize=21, leading=30, textColor=INK),
    "cover_h2": st("cover_h2", fontName=FONT_B, fontSize=17, leading=24, textColor=INK),
    "cover_meta": st("cover_meta", fontName=FONT, fontSize=11, leading=16, textColor=MUTED),
    "cover_en": st("cover_en", fontName=FONT_EN, fontSize=10.5, leading=15, textColor=MUTED, alignment=TA_LEFT),
    "h1": st("h1", fontName=FONT_B, fontSize=20, leading=28, textColor=INK),
    "h2": st("h2", fontName=FONT_B, fontSize=13, leading=18, textColor=GREEN),
    "body": st("body", alignment=TA_LEFT, spaceAfter=6),
    "body_b": st("body_b", fontName=FONT_B, alignment=TA_LEFT, spaceAfter=6),
    "quote": st("quote", fontSize=11, leading=18, alignment=TA_LEFT),
    "stat_n": st("stat_n", fontName=FONT_B, fontSize=18, leading=22, textColor=INK),
    "stat_l": st("stat_l", fontSize=8.5, leading=12, textColor=MUTED),
    "cap": st("cap", fontSize=8.5, leading=12, textColor=MUTED, spaceBefore=4, spaceAfter=8),
    "note": st("note", fontSize=8.5, leading=13, textColor=MUTED, alignment=TA_LEFT),
    "card_t": st("card_t", fontName=FONT_B, fontSize=10.5, leading=14, textColor=ACCENT, spaceAfter=4),
    "card_b": st("card_b", fontSize=10, leading=15.5, alignment=TA_LEFT),
    "th": st("th", fontName=FONT_B, fontSize=8, leading=11, textColor=colors.white, alignment=TA_CENTER),
    "td": st("td", fontSize=8.5, leading=11, alignment=TA_CENTER),
    "td_l": st("td_l", fontSize=8.5, leading=11, alignment=TA_LEFT),
    "td_b": st("td_b", fontName=FONT_B, fontSize=8.5, leading=11, alignment=TA_CENTER),
    "m_tag": st("m_tag", fontName=FONT_B, fontSize=11, leading=14, textColor=GREEN, alignment=TA_CENTER),
    "m_name": st("m_name", fontName=FONT_B, fontSize=10, leading=13, alignment=TA_CENTER),
    "m_body": st("m_body", fontSize=8.5, leading=12.5, alignment=TA_LEFT),
    "box_l": st("box_l", fontName=FONT_B, fontSize=9, leading=12, textColor=GREEN_DARK, alignment=TA_CENTER),
    "box_s": st("box_s", fontSize=7.5, leading=10.5, textColor=MUTED, alignment=TA_CENTER),
    "flow": st("flow", fontName=FONT_B, fontSize=10, leading=13, textColor=colors.white, alignment=TA_CENTER),
    "bullet_h": st("bullet_h", fontName=FONT_B, fontSize=10.5, leading=16, alignment=TA_LEFT),
}


class Doc(BaseDocTemplate):
    # page 1 = cover; body sections in order
    SECTION_TITLES = [
        "",  # cover
        "1. 科学问题（一句话）",
        "2. 核心思想",
        "3. 模型演化：每一步解决一个具体的生态/物理问题",
        "4. 当前结果（关键数字）",
        "5. 当前发现 与 待讨论的问题",
    ]

    def __init__(self, path):
        super().__init__(
            path, pagesize=A4,
            leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
            title="river_graph", author="river_graph",
        )
        frame = Frame(ML, MB, CW, H - MT - MB, id="n",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=self._cover),
            PageTemplate(id="body", frames=[frame], onPage=self._body),
        ])

    def _cover(self, canv, doc):
        canv.saveState()
        canv.setFillColor(BG)
        canv.rect(0, 0, W, H, fill=1, stroke=0)
        canv.setFillColor(GREEN)
        canv.rect(0, H - 5 * mm, W, 5 * mm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#D9E7E0"))
        canv.rect(0, 28 * mm, W, 4 * mm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#C5D9CE"))
        canv.rect(0, 22 * mm, W, 6 * mm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#A8C4B5"))
        canv.rect(0, 16 * mm, W, 6 * mm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor("#2F7A5F"))
        canv.rect(0, 0, W, 16 * mm, fill=1, stroke=0)
        canv.restoreState()

    def _body(self, canv, doc):
        canv.saveState()
        canv.setFillColor(BG)
        canv.rect(0, 0, W, H, fill=1, stroke=0)
        page = canv.getPageNumber()
        titles = self.SECTION_TITLES
        title = titles[page - 1] if 0 < page <= len(titles) else (titles[-1] if titles else "")
        if title:
            canv.setFont(FONT, 8)
            canv.setFillColor(MUTED)
            canv.drawString(ML, H - MT + 8, title)
        canv.setFont(FONT, 8)
        canv.setFillColor(MUTED)
        canv.drawString(ML, 10 * mm, str(doc.page))
        canv.restoreState()


class SectionMark(Spacer):
    """Kept for story clarity; titles come from Doc.SECTION_TITLES by page."""

    def __init__(self, title):
        super().__init__(1, 0.01)
        self.title = title


def sec_h(text):
    t = Table([[Paragraph(text, S["h1"])]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 1.3, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def quote_box(html):
    t = Table([[Paragraph(html, S["quote"])]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), GREEN_LIGHT),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, GREEN),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def accent_card(title, body):
    inner = Table(
        [[Paragraph(title, S["card_t"])], [Paragraph(body, S["card_b"])]],
        colWidths=[CW - 4],
    )
    inner.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    t = Table([[inner]], colWidths=[CW])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 3.5, ACCENT_BAR),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def stats_row():
    cw = CW / 3
    items = [
        ("11,948", "密西西比河流域内至少有一次<br/>DOC 记录的 USGS 站点"),
        ("约 570", "其中有可用记录、进入图建模的<br/>站点"),
        ("8.9%", "（站点 × 月）网格中真正被观测<br/>的比例"),
    ]
    cells = []
    for num, lab in items:
        stack = Table(
            [[Paragraph(num, S["stat_n"])], [Spacer(1, 3)], [Paragraph(lab, S["stat_l"])]],
            colWidths=[cw - 16],
        )
        stack.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        cells.append(stack)
    t = Table([cells], colWidths=[cw, cw, cw])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBEFORE", (0, 0), (0, 0), 2.2, GREEN),
        ("LINEBEFORE", (1, 0), (1, 0), 1.2, LINE),
        ("LINEBEFORE", (2, 0), (2, 0), 1.2, LINE),
    ]
    t.setStyle(TableStyle(style))
    return t


def flow_diagram():
    """Core idea flow: 3 inputs → HydroGRN → DOC output."""
    bw, bh = 118, 52
    # Use nested table for layout
    def box(title, lines, bg=colors.HexColor("#F4F6F4"), fg=INK, bold_line=0):
        parts = [Paragraph(f"<b>{title}</b>" if bold_line == 0 else title,
                           st("bt", fontName=FONT_B, fontSize=9, leading=12, textColor=fg, alignment=TA_CENTER))]
        for ln in lines:
            parts.append(Paragraph(ln, st("bl", fontSize=8, leading=11, textColor=fg, alignment=TA_CENTER)))
        stack = Table([[p] for p in parts], colWidths=[bw - 4])
        stack.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        outer = Table([[stack]], colWidths=[bw], rowHeights=[bh])
        outer.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("BOX", (0, 0), (-1, -1), 0.6, LINE),
            ("LEFTPADDING", (0, 0), (-1, -1), 2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        return outer

    def arrow():
        a = Table([[Paragraph("→", st("ar", fontSize=14, leading=16, textColor=GREEN, alignment=TA_CENTER))]],
                  colWidths=[18], rowHeights=[bh])
        a.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return a

    eco = box("流域生态背景", ["土壤 · 地形 · 水文 · 气候", "StreamCat 集水区属性"], bg=GREEN_LIGHT)
    obs = box("稀疏 DOC 观测", ["8.9% 站点月有值"])
    graph = box("河网图", ["连通性 · 流向 · 输移属性", "571 站 / 562 有向边"])
    model = box("HydroGRN", ["有向 + 输移门控 GNN", "+ 生态背景编码器"], bg=colors.HexColor("#E8F0EC"))
    out = box("DOC 全时空重建", ["571 站 × 652 个月"], bg=GREEN_LIGHT)

    # Layout: left column eco / (obs+graph), middle model, right out
    left = Table(
        [
            [eco, ""],
            [Spacer(1, 10), ""],
            [Table([[obs]], colWidths=[bw], rowHeights=[44]), Table([[graph]], colWidths=[bw], rowHeights=[52])],
        ],
        colWidths=[bw + 4, bw],
    )
    # simpler horizontal-ish diagram with 3 rows
    row1 = Table([[eco]], colWidths=[bw + 30], rowHeights=[bh])
    row1.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Final compact diagram as a single figure via canvas-like table
    d = Table(
        [
            ["", eco, "", model, out],
            [obs, graph, "", "", ""],
        ],
        colWidths=[bw + 6, bw + 6, 16, bw + 6, bw + 6],
        rowHeights=[bh + 4, bh],
    )
    d.setStyle(TableStyle([
        ("SPAN", (1, 0), (1, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        # arrows as text placeholders via extra cells - use LINE draw
        ("BOX", (0, 0), (-1, -1), 0, colors.white),
    ]))
    # Cleaner approach: one row of 5 boxes
    d2 = Table(
        [[obs, arrow(), graph, arrow(), model, arrow(), out]],
        colWidths=[bw + 4, 16, bw + 4, 16, bw + 8, 16, bw + 4],
    )
    d2.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
    ]))
    # Put eco above model with a note row
    eco_row = Table(
        [["", "", "", "", eco, "", ""]],
        colWidths=[bw + 4, 16, bw + 4, 16, bw + 8, 16, bw + 4],
        rowHeights=[bh],
    )
    eco_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (4, 0), (4, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 1),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    # arrows into model from eco
    arrow_in = Table(
        [["", "", "", "", Paragraph("↘", st("a2", fontSize=12, textColor=GREEN, alignment=TA_CENTER)), "", ""]],
        colWidths=[bw + 4, 16, bw + 4, 16, bw + 8, 16, bw + 4],
        rowHeights=[14],
    )
    arrow_in.setStyle(TableStyle([
        ("ALIGN", (4, 0), (4, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    wrap = Table([[eco_row], [arrow_in], [d2]], colWidths=[CW])
    wrap.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return wrap


def model_ladder():
    rows = [
        ("G0", "普通 GCN",
         "检验：<b>河流是不是普通空间？</b>把河网当无向图做邻居平均。结论：真实河网 &gt; 随机图 &gt; 无图（E1），且只有真实河网能在 E3 空间外推中迁移——拓扑本身是有效信息。"),
        ("H1", "有向关系 GCN",
         "补上：<b>上下游方向。</b>上游、下游两个方向使用独立的权重通道（上游是因果输入，下游是反向约束）。是第一个在 E2 时间外推上击败随机森林的模型。"),
        ("H2", "输移门控 GCN",
         "补上：<b>输移强度的差异。</b>边不再是等权重的——由跳数距离、河长、汇水面积、坡度、河级经 MLP 生成逐边门控权重，“河段输移”代替“均匀平均”。E2a/E2b 最优，E1 追平随机森林。"),
        ("H2E+", "生态背景",
         "补上：<b>生态异质性。</b>把 StreamCat 集水区属性（土地利用、气候、土壤有机质、高程、基流）作为节点特征。修复了此前最差的空间迁移：E3 R² 从 0.23 → 0.32。"),
        ("H2X+", "生态编码器",
         "把生态背景从“原始特征”升级为“学习到的嵌入”（MLP 编码生态属性块，再与图表示融合）。当前最优模型：<b>E2b MAE 0.96 / R² 0.47；E3 最优种子 R² 0.53</b>。"),
    ]
    data = []
    for tag, name, desc in rows:
        data.append([
            Paragraph(tag, S["m_tag"]),
            Paragraph(name, S["m_name"]),
            Paragraph(desc, S["m_body"]),
        ])
    t = Table(data, colWidths=[16 * mm, 30 * mm, CW - 46 * mm])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (0, -1), GREEN_LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.5, LINE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("BACKGROUND", (0, -1), (0, -1), GREEN),
        ("TEXTCOLOR", (0, -1), (0, -1), colors.white),
    ]
    # last row tag already white-on-green via style override on paragraph? redraw last tag
    t.setStyle(TableStyle(style))
    return t


def results_table():
    headers = [
        Paragraph("模型", S["th"]),
        Paragraph("E1 随机掩码<br/>r20", S["th"]),
        Paragraph("E2a 严格未来<br/>预测", S["th"]),
        Paragraph("E2b 未来重建<br/>（监测网运行中）", S["th"]),
        Paragraph("E3 空间外推<br/>（整站留出）", S["th"]),
    ]
    # From pypdfium2 extraction of original page 5:
    # 站点均值 | 1.46 | 1.14 | 1.12 | 2.59±0.45
    # 克里金 | 2.15 | — | 1.51 | 2.05±0.29
    # 随机森林 | 1.42 | 1.11 | 1.11 | 2.07±0.20
    # G0 | 1.74 | 1.45 | 1.38 | 1.97±0.31
    # H1 | 1.50 | 1.09 | 1.07 | 2.03±0.23
    # H2 | 1.41 | 1.10 | 1.05 | 2.01±0.26
    # H2E | 1.41 | 1.18 | 1.13 | 1.81±0.21
    # H2X | 1.40 | — | 0.96 | 1.79±0.28
    raw = [
        ("站点均值", "1.46", "1.14", "1.12", "2.59 ± 0.45", False),
        ("克里金（欧氏空间）", "2.15", "—", "1.51", "2.05 ± 0.29", False),
        ("随机森林", "1.42", "1.11", "1.11", "2.07 ± 0.20", False),
        ("G0 普通 GCN（河网图）", "1.74", "1.45", "1.38", "1.97 ± 0.31", False),
        ("H1 有向河网", "1.50", "1.09", "1.07", "2.03 ± 0.23", False),
        ("H2 输移门控", "1.41", "1.10", "1.05", "2.01 ± 0.26", False),
        ("H2E ＋生态背景", "1.41", "1.18", "1.13", "1.81 ± 0.21", False),
        ("H2X ＋生态编码器（当前最优）", "1.40", "—", "0.96", "1.79 ± 0.28", True),
    ]
    data = [headers]
    for name, a, b, c, d, highlight in raw:
        style = S["td_b"] if highlight else S["td"]
        data.append([
            Paragraph(name, S["td_l"]),
            Paragraph(a, style),
            Paragraph(b, style),
            Paragraph(c, style),
            Paragraph(d, style),
        ])
    t = Table(data, colWidths=[CW * 0.28, CW * 0.155, CW * 0.155, CW * 0.19, CW * 0.22], repeatRows=1)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), GREEN),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F3F6F4")),
        ("BACKGROUND", (0, -1), (-1, -1), GREEN_LIGHT),
    ]
    t.setStyle(TableStyle(style_cmds))
    return t


def bullet_line(lead, rest):
    return Paragraph(f"• <b>{lead}</b>——{rest}", S["body"])


def build():
    story = []
    doc = Doc(str(OUT))

    # ===== COVER =====
    story.append(NextPageTemplate("body"))
    story.append(Spacer(1, 52 * mm))
    story.append(Paragraph("RIVER_GRAPH", S["brand"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("面向稀疏观测的河流溶解性有机碳（DOC）重建", S["cover_h1"]))
    story.append(Paragraph("——基于河网拓扑与生态属性的图神经网络", S["cover_h2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("密西西比河流域 · 571 个监测站点 · 1972–2026 逐月 DOC", S["cover_meta"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Topology-aware graph neural networks for reconstructing sparse DOC observations "
        "in the Mississippi River Basin",
        S["cover_en"],
    ))
    # NOTE: intentionally omit 汇报人/汇报对象/日期/代码仓库 block
    story.append(PageBreak())

    # ===== §1 =====
    story.append(SectionMark("1. 科学问题（一句话）"))
    story.append(sec_h("1. 科学问题（一句话）"))
    story.append(Spacer(1, 8))
    story.append(quote_box(
        "河流碳监测在空间和时间上都高度稀疏。能否利用<b>河网本身的拓扑结构</b>（连通性、流向、输移物理过程）"
        "和<b>流域生态属性</b>（土壤、地形、水文、气候），恢复从未被观测的 DOC 状态，"
        "从而降低流域碳通量估计的不确定性？"
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph("这个问题的现实背景，是监测数据本身的极度稀疏：", S["body"]))
    story.append(stats_row())
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "换句话说，现有碳通量估计所依赖的观测底账里，<b>超过九成的“站点-月份”格子是空的</b>。"
        "传统的填补方式是把河流当作普通的空间点集（欧氏距离克里金、随机森林按特征外推），"
        "但河流不是普通空间——它是有方向的输移系统，上游站点对下游是因果输入，"
        "下游观测对上游是反向约束，而且每条河段的输移强度由河长、汇水面积、坡度、河级等物理属性决定。"
        "本项目的工作假设是：这些结构与生态信息，本身就携带了恢复缺失观测所需的信号。",
        S["body"],
    ))
    story.append(Spacer(1, 6))
    if MAP_PATH.exists():
        img = Image(str(MAP_PATH))
        max_w = CW * 0.72
        max_h = 150
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_w / iw, max_h / ih)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Paragraph(
            "图 1 密西西比河流域 571 个 DOC 监测站点示意（按图连通分量着色）。"
            "示意性渲染图，基于真实图统计（571 节点 / 562 条有向边）；管线等价图由 build_graph.py 生成。",
            S["cap"],
        ))
    story.append(PageBreak())

    # ===== §2 =====
    story.append(SectionMark("2. 核心思想"))
    story.append(sec_h("2. 核心思想"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "把“稀疏站点上的零散观测”放进河网结构与流域生态背景中传播，是整条方法链的核心：",
        S["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(flow_diagram())
    story.append(Spacer(1, 10))
    story.append(Paragraph("三个输入各自回答一个“普通空间方法回答不了”的问题：", S["body"]))
    story.append(bullet_line(
        "稀疏观测",
        "每个月只有少数站点有实测值，模型输入包含“已观测 DOC 通道 + 掩码”，让已有观测沿图传播到同月的空白站点；",
    ))
    story.append(bullet_line(
        "河网图",
        "通过 NLDI / NHDPlus 构建的有向图（上游 → 下游），每条边携带跳数距离、河长、汇水面积、坡度、河级等输移属性；",
    ))
    story.append(bullet_line(
        "生态背景",
        "每个站点所在集水区的土地利用、气候多年平均、土壤有机质、高程、基流指数（EPA StreamCat），刻画“这个站点为什么是这个碳性格”。",
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "预测的单元是一个（站点，月）格子；DOC 在对数空间建模，指标换回 mg/L 报告。"
        "基准实验覆盖三种外推情形：<b>E1</b> 随机掩码（20/40/60%）、<b>E2a</b> 严格未来预测（2021 年后无任何观测）、"
        "<b>E2b</b> 监测网持续运行下的未来重建（2021 年后保留 20% 站点作为上下文，最接近真实业务场景）、"
        "<b>E3</b> 空间外推（整站留出 20%，检验“信息能否沿河网传到从未监测的站点”）。",
        S["body"],
    ))
    story.append(PageBreak())

    # ===== §3 =====
    story.append(SectionMark("3. 模型演化：每一步解决一个具体的生态/物理问题"))
    story.append(sec_h("3. 模型演化：每一步解决一个具体的生态/物理问题"))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "模型不是一步到位堆出来的，而是在<b>同一套冻结基准</b>上逐级消融演化。每一级只改一件事，"
        "因此每一级的增益都能归因到具体的机制假设：",
        S["body"],
    ))
    story.append(Spacer(1, 6))
    story.append(model_ladder())
    story.append(Paragraph(
        "注：演化阶梯中每一级都在同一组掩码、同一泄漏控制规则、同一早停协议下重跑，"
        "消融结论可复现（掩码与冻结结果存于 experiments/masks 与 experiments/frozen_results）。",
        S["note"],
    ))
    story.append(PageBreak())

    # ===== §4 =====
    story.append(SectionMark("4. 当前结果（关键数字）"))
    story.append(sec_h("4. 当前结果（关键数字）"))
    story.append(Spacer(1, 8))
    story.append(Paragraph("4.1 主结果：三类外推场景（MAE mg/L，越低越好）", S["h2"]))
    story.append(Paragraph(
        "表 1 关键模型在三类基准上的表现（E1 为三种子均值 ± 标准差；E3 为三种子均值 ± 标准差；"
        "数据来自 experiments/results/summary.md）",
        S["note"],
    ))
    story.append(results_table())
    story.append(Paragraph(
        "E2a 上克里金无定义（测试月没有任何同期观测可插值）；H2X 的 E1 r40/r60 与 E2a 冻结批次仍在补跑，"
        "权威完整表见仓库 experiments/frozen_results/benchmark.csv。"
        "R² 口径：E2b 上 H2X 为 0.47（RF 为 0.18），E3 上最优种子达 0.53。",
        S["note"],
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("4.2 读法：三个场景分别说明什么", S["h2"]))
    story.append(Paragraph(
        "<b>E2b（最贴近真实业务）</b>：监测网持续运行、少数站点照常上报时，H2X 把 MAE 压到 0.96 mg/L，"
        "比随机森林（1.11）低约 14%，R² 从 0.18 提到 0.47——河网传播 + 生态编码带来的增益主要落在这里；",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>E3（空间外推）</b>：所有方法都吃力（留出站的站点均值都到 2.59），但 GNN 家族整体优于空间插值与随机森林；"
        "生态背景把 R² 从 0.23 修复到 0.32，说明“这个站是什么生态性格”是空间迁移的关键缺失变量；",
        S["body"],
    ))
    story.append(Paragraph(
        "<b>E1（随机掩码）</b>：在轻掩码（20%）下 H2X 已略优于随机森林；60% 重掩码区间随机森林仍占优"
        "——图传播在观测极度稀薄时也会“断粮”。",
        S["body"],
    ))
    story.append(PageBreak())

    # ===== §5 — remove 老师 wording =====
    # Original: "5. 当前发现 与 想请教老师的问题"
    # Original: "5.3 想请老师指导的三个问题"
    # Original: "想听老师的判断"
    sec5 = "5. 当前发现 与 待讨论的问题"
    story.append(SectionMark(sec5))
    story.append(sec_h(sec5))
    story.append(Spacer(1, 8))
    story.append(Paragraph("5.1 两个值得讨论的发现", S["h2"]))
    story.append(accent_card(
        "发现一：E3 的失败集中在源头小流域，而生态背景恰好能救回它们。",
        "对 E3 误差的逐站归因显示，最大的失败来自密苏里河源头区的源头站点——这些站点与下游主干"
        "的“图距离”远、且水文气候背景与大河谷站点差异极大。加入 StreamCat 生态背景后，这些站点的 "
        "MAE 从 2.79 降到 2.21。这提示：空间迁移的瓶颈不是“图传得不够远”，"
        "而是<b>模型不知道远端站点处于另一种生态状态</b>。",
    ))
    story.append(Spacer(1, 8))
    story.append(accent_card(
        "发现二：极端 DOC 状态是系统性短板，而且它的成因“看不见”。",
        "以 06438000 站为例的标签审计：该站 1978–1980 年出现一轮月均 DOC 高达 137.5 mg/L 的局部高值期"
        "（峰值 460，1980-07），1981 年后回落至个位数；同期流量与 DOC 的 Spearman 相关为 −0.00，"
        "上游站点全程干净——这是<b>一段局部的、历史性的源区/土地利用或采样年代事件</b>。"
        "在只有月均流量、没有事件与源区历史状态的条件下，任何空间模型都没有依据外推出 460 这样的值。"
        "系统性低估极端值是可解释的科学局限，不是单纯的工程失败。",
    ))
    story.append(Spacer(1, 6))
    story.append(Paragraph("5.2 当前判断与下一步设想", S["h2"]))
    story.append(Paragraph(
        "当前结果显示，河网拓扑与生态属性能显著提升 DOC 重建的<b>时间外推（E2b）</b>与"
        "<b>空间迁移（E3）</b>能力，但在<b>极端 DOC 状态</b>下仍存在系统性不足。结合发现二，最可能的缺口是："
        "<b>模型没有时间状态</b>——月均标签下的输移记忆、前期湿润/干旱过程、事件性冲刷都不可见。",
        S["body"],
    ))
    story.append(Paragraph(
        "下一步计划是在 H2X 基础上加入<b>逐月时序建模</b>（每个站点的图嵌入序列上接 GRU 类结构，"
        "暂名 <b>HydroGRN-ST</b>；设计文档中对应消融阶梯的 H3 级），"
        "并在 E2a/E2b 上检验时间记忆能否补上极端状态的缺口。",
        S["body"],
    ))
    # Original 5.3 title: 想请老师指导的三个问题 → 待讨论的三个问题
    story.append(Paragraph("5.3 待讨论的三个问题", S["h2"]))
    story.append(accent_card(
        "问题一（方向层面）：",
        "加入时序建模（HydroGRN-ST）以恢复极端 DOC 状态，这个思路是否符合碳输移的生态过程认识？"
        "从生态学角度，月均分辨率下“前期过程记忆”对 DOC 的约束有多强？是否应优先做事件尺度而不是月尺度？",
    ))
    story.append(Spacer(1, 8))
    story.append(accent_card(
        "问题二（机制层面）：",
        "发现一显示空间迁移瓶颈在“生态状态差异”而非“图距离”。除 StreamCat 现有的土壤/地形/气候/基流变量外，"
        "从河流碳循环角度，还应优先补充哪类集水区属性（如湿地比例、多年冻土、农业强度、植被生产力）？",
    ))
    story.append(Spacer(1, 8))
    # Original ending: 想听老师的判断 → 想进一步明确判断
    story.append(accent_card(
        "问题三（数据层面）：",
        "06438000 这类“历史局部高值期”标签，在建模上应作为真实但不可外推的状态保留（并在不确定度中体现），"
        "还是作为年代/方法效应的异常处理？这对碳通量总量估计的口径有影响，想进一步明确判断。",
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "材料数据口径：全部指标取自仓库冻结结果（experiments/frozen_results/benchmark.csv 与 "
        "experiments/results/summary.md）；图结构为 571 站 / 562 条有向边（NLDI over NHDPlus，"
        "最大连通分量 407 节点）；标签为 USGS pcode 00681 月均 DOC，1972-04 至 2026-07 共 33,048 个已观测"
        "（站点，月）格子；复现管线与数据来源（USGS NWIS、WQP、NLDI/NHDPlus、EPA StreamCat）见仓库 README。",
        S["note"],
    ))

    doc.build(story)
    print("wrote", OUT, "size", OUT.stat().st_size)


if __name__ == "__main__":
    build()
