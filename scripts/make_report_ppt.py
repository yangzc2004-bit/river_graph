"""Academic progress-report deck for river_graph DOC reconstruction."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path(r"D:\river_graph\阶段成果汇报_河网GNN_DOC重构.pptx")

# palette
BG = RGBColor(0xF7, 0xF6, 0xF2)
INK = RGBColor(0x1A, 0x1A, 0x1A)
NAVY = RGBColor(0x1B, 0x3A, 0x5C)
GREEN = RGBColor(0x2C, 0x6E, 0x49)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
LINE = RGBColor(0xC9, 0xC5, 0xBC)
FILL = RGBColor(0xE8, 0xE4, 0xDC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0x8B, 0x2E, 0x2E)

CN = "Microsoft YaHei"
EN = "Calibri"


def set_cjk(run, name=CN):
    run.font.name = name
    rPr = run._r.get_or_add_rPr()
    successors = {
        "a:ea": ("a:cs", "a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
        "a:cs": ("a:sym", "a:hlinkClick", "a:hlinkMouseOver", "a:rtl", "a:extLst"),
    }
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.insert_element_before(el, *successors[tag])
        el.set("typeface", name)


def style_run(run, size=16, bold=False, color=INK, name=CN):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    set_cjk(run, name)


def add_bg(slide, prs):
    r = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
    )
    r.fill.solid()
    r.fill.fore_color.rgb = BG
    r.line.fill.background()
    r.shadow.inherit = False
    return r


def add_title_bar(slide, prs, title, subtitle=None):
    # navy accent bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.38), Inches(0.12), Inches(0.55)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(0.85), Inches(0.32), Inches(11.8), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    style_run(run, 26, True, NAVY)

    if subtitle:
        st = slide.shapes.add_textbox(
            Inches(0.85), Inches(0.88), Inches(11.8), Inches(0.32)
        )
        p2 = st.text_frame.paragraphs[0]
        r2 = p2.add_run()
        r2.text = subtitle
        style_run(r2, 12, False, GRAY)

    # thin divider
    ln = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.22), Inches(12.2), Inches(0.015)
    )
    ln.fill.solid()
    ln.fill.fore_color.rgb = LINE
    ln.line.fill.background()
    ln.shadow.inherit = False


def add_footer(slide, page, total=18):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(7.1), Inches(12.2), Inches(0.28))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = f"河网感知 GNN 重构稀疏 DOC 观测  ·  阶段成果汇报  ·  {page}/{total}"
    style_run(r, 10, False, GRAY)


def add_textbox(slide, left, top, width, height, paragraphs, size=16, color=INK, bold=False):
    """paragraphs: list of str or (str, size, bold, color)"""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        if isinstance(item, tuple):
            text, sz, bd, col = item
        else:
            text, sz, bd, col = item, size, bold, color
        run = p.add_run()
        run.text = text
        style_run(run, sz, bd, col)
    return tb


def add_rect(slide, left, top, width, height, fill, line=None):
    r = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    if line is None:
        r.line.fill.background()
    else:
        r.line.color.rgb = line
        r.line.width = Pt(1)
    r.shadow.inherit = False
    # less rounded
    try:
        r.adjustments[0] = 0.08
    except Exception:
        pass
    return r


def add_arrow(slide, left, top, width, height=0.02):
    r = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    r.fill.solid()
    r.fill.fore_color.rgb = NAVY
    r.line.fill.background()
    r.shadow.inherit = False
    return r


def add_table(slide, left, top, width, rows, col_widths=None, header=True):
    """rows: list of list of str"""
    n_rows = len(rows)
    n_cols = len(rows[0])
    table_shape = slide.shapes.add_table(
        n_rows, n_cols, Inches(left), Inches(top), Inches(width), Inches(0.38 * n_rows)
    )
    table = table_shape.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            table.columns[i].width = Emu(int(Inches(width) * w / total))

    for r_i, row in enumerate(rows):
        for c_i, cell_text in enumerate(row):
            cell = table.cell(r_i, c_i)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_i > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(cell_text)
            is_header = header and r_i == 0
            is_ours = (not is_header) and any(
                k in str(row[0]) for k in ("H2X", "H2E", "H2", "H1", "G0")
            ) and c_i == 0
            style_run(
                run,
                12 if not is_header else 12,
                is_header or is_ours,
                WHITE if is_header else (GREEN if is_ours else INK),
            )
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            # fill
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            solid = tcPr.makeelement(qn("a:solidFill"), {})
            srgb = solid.makeelement(qn("a:srgbClr"), {"val": "1B3A5C" if is_header else ("EEF3EE" if is_ours and c_i == 0 else "FFFFFF")})
            solid.append(srgb)
            # remove existing solidFill
            for child in list(tcPr):
                if child.tag == qn("a:solidFill"):
                    tcPr.remove(child)
            tcPr.append(solid)
    return table_shape


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(s, prs)
    return s


FIG_DIR = Path(r"D:\river_graph\docs\figures")


def add_figure_slide(prs, title, image_name, caption, note, page):
    """Full-width academic figure slide."""
    s = blank(prs)
    add_title_bar(s, prs, title)
    img = FIG_DIR / image_name
    # source is ~1491x1055 → fit into content box while keeping aspect
    max_w, max_h = 11.8, 5.35
    from PIL import Image

    with Image.open(img) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    left = (13.333 - w) / 2
    top = 1.42 + (max_h - h) / 2
    s.shapes.add_picture(str(img), Inches(left), Inches(top), Inches(w), Inches(h))
    add_textbox(s, 0.55, 6.85, 12.2, 0.28, [(caption, 11, False, GRAY)])
    add_footer(s, page)
    notes(s, note)
    return s


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ── 1 Cover ──────────────────────────────────────────────
    s = blank(prs)
    accent = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.22), prs.slide_height
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = NAVY
    accent.line.fill.background()
    accent.shadow.inherit = False

    add_textbox(
        s, 0.9, 1.6, 11.5, 0.4,
        [("阶段成果汇报", 14, False, GRAY)],
    )
    add_textbox(
        s, 0.9, 2.1, 11.5, 1.4,
        [("河网感知图神经网络", 34, True, NAVY),
         ("重构密西西比河流域稀疏 DOC 观测", 28, True, NAVY)],
    )
    add_textbox(
        s, 0.9, 3.8, 11.5, 1.2,
        [("Topology-aware GNN for Reconstructing Sparse Dissolved Organic Carbon", 14, False, GRAY),
         ("Observations in the Mississippi River Basin", 14, False, GRAY)],
    )
    # meta line
    meta = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(5.3), Inches(4.2), Inches(0.04)
    )
    meta.fill.solid()
    meta.fill.fore_color.rgb = NAVY
    meta.line.fill.background()
    meta.shadow.inherit = False
    add_textbox(
        s, 0.9, 5.55, 11, 0.8,
        [("科研进展 / 方法阶梯已闭合 / 冻结与论文图进行中", 14, False, INK),
         ("2026-09", 12, False, GRAY)],
    )
    notes(s, "开场：这是河网GNN做DOC重构的阶段汇报。方法主线已经闭合，正在做结果冻结与论文图。")

    # ── 2 Motivation ─────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "研究背景：DOC 监测极稀疏", "溶解有机碳（DOC）是水质与碳循环的关键指标，但观测网络远不足以支撑全流域理解")
    # three callouts
    stats = [
        ("11,948", "至少有过 1 次\nDOC 采样的站点", NAVY),
        ("~570", "具有可用时间序列\n的站点", NAVY),
        ("8.9%", "(站点×月份) 格子\n实际有观测", GREEN),
    ]
    for i, (num, label, col) in enumerate(stats):
        x = 0.7 + i * 4.15
        box = add_rect(s, x, 1.6, 3.85, 2.3, WHITE, LINE)
        add_textbox(s, x + 0.2, 1.85, 3.45, 1.0, [(num, 40, True, col)])
        add_textbox(s, x + 0.2, 2.95, 3.45, 0.8, [(label, 14, False, GRAY)])

    add_textbox(
        s, 0.7, 4.2, 12, 2.4,
        [
            ("传统路径的局限", 16, True, NAVY),
            ("· 站均值 / 空间插值（Kriging）：忽略河网连通与水流方向，空间外推弱", 15, False, INK),
            ("· 表格特征模型（RF / MLP）：把站点当独立点，未利用上下游因果结构", 15, False, INK),
            ("· 普通 GCN：把邻域当无向平均，与河流「有向输运」物理不符", 15, False, INK),
            ("→ 核心矛盾：观测稀疏 vs 河网本身携带强结构先验", 15, True, GREEN),
        ],
    )
    add_footer(s, 2)
    notes(s, "用三个数字建立问题规模：上万站有过采样，可用序列只有约五百，覆盖率8.9%。再点出三种常规方法为什么不够。")

    # ── 3 Scientific question ────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "科学问题")
    # big quote-like box
    qbox = add_rect(s, 0.7, 1.55, 12, 1.6, WHITE, NAVY)
    add_textbox(
        s, 1.0, 1.75, 11.4, 1.3,
        [("能否利用河网自身的连通性、流向、输运物理与生态身份，", 18, True, NAVY),
         ("重构那些「从未被测量」的 (站点, 月份) DOC？", 18, True, NAVY)],
    )
    rqs = [
        ("RQ1", "真实河网拓扑是否优于随机图 / 无图？"),
        ("RQ2", "在 20/40/60% 重遮罩下能恢复多少观测？"),
        ("RQ3", "学到的结构能否在时间与空间上外推？"),
    ]
    for i, (tag, text) in enumerate(rqs):
        y = 3.5 + i * 0.95
        tag_box = add_rect(s, 0.7, y, 1.1, 0.65, NAVY)
        add_textbox(s, 0.8, y + 0.12, 0.9, 0.45, [(tag, 16, True, WHITE)])
        add_textbox(s, 2.0, y + 0.12, 10.5, 0.55, [(text, 16, False, INK)])

    add_textbox(
        s, 0.7, 6.4, 12, 0.4,
        [("任务形态：站点图上的时空遮罩重构（spatiotemporal masked reconstruction）", 13, False, GRAY)],
    )
    add_footer(s, 3)
    notes(s, "把问题收成一句科学问题，再拆成RQ1拓扑、RQ2缺失恢复、RQ3时空泛化三个可证伪子问题。")

    add_figure_slide(
        prs,
        "Figure 1 · 科学问题示意",
        "fig1_scientific_question.png",
        "稀疏 DOC 观测 vs 欧氏插值 vs 拓扑感知 GNN：关键在于河网连通是否提供超出距离的信息。",
        "对应Fig1：左边稀疏观测，中间距离插值忽略流向，右边沿河网消息传递补全缺失DOC。",
        4,
    )

    # ── Key idea ─────────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "核心思路", "环境定义节点「是什么」，河网定义信息「怎么传」")
    # pipeline boxes
    boxes = [
        (0.55, "稀疏 DOC 观测", "USGS / WQP\n8.9% cells"),
        (2.95, "河网拓扑", "NLDI / NHDPlus\n有向边 + 物理属性"),
        (5.35, "生态身份", "StreamCat\n汇水区静态属性"),
        (7.75, "有向传输 GNN", "direction + gate\n+ context encoder"),
    ]
    for x, title, sub in boxes:
        add_rect(s, x, 1.7, 2.2, 1.9, WHITE, LINE)
        add_textbox(s, x + 0.12, 1.9, 1.95, 0.9, [(title, 15, True, NAVY)])
        add_textbox(s, x + 0.12, 2.85, 1.95, 0.6, [(sub, 11, False, GRAY)])
    for x in (2.72, 5.12, 7.52):
        add_arrow(s, x, 2.5, 0.22, 0.28)

    # result callout — same width as peers
    add_rect(s, 10.15, 1.7, 2.2, 1.9, GREEN)
    add_textbox(s, 10.3, 2.05, 1.95, 1.2, [("补全未测\nDOC", 16, True, WHITE)])

    add_textbox(
        s, 0.7, 4.0, 12, 2.6,
        [
            ("设计原则", 16, True, NAVY),
            ("1. 拓扑消融：无图 / 随机图 / 真实河网 —— 证明「河网」本身有效", 15, False, INK),
            ("2. 方向消融：无向 GCN → 上/下游分离权重 —— 证明「流向」有效", 15, False, INK),
            ("3. 传输消融：均匀邻域平均 → 边物理门控 —— 证明「河段属性」有效", 15, False, INK),
            ("4. 身份消融：仅拓扑 → +生态 context —— 证明空间外推需要「这是什么河」", 15, False, INK),
            ("每一级只改一个因素，同一 benchmark 协议对比", 14, True, GREEN),
        ],
    )
    add_footer(s, 5)
    notes(s, "讲清输入三件套和输出目标。强调消融阶梯：每一步只加一个物理/生态假设。")

    add_figure_slide(
        prs,
        "Figure 2 · 实验框架总览",
        "fig2_experimental_framework.png",
        "Data → Graph construction → Methods → Evaluation → Outputs：MVP 全链路。",
        "对应Fig2：从数据到构图、基线与GNN、E1/E2/E3评测、重构输出与碳通量含义。",
        6,
    )

    # ── Data pipeline ────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "数据与图构建", "密西西比河流域 · 月尺度 · 公开数据可复现")
    rows = [
        ["组件", "来源", "规模 / 内容"],
        ["DOC 标签", "USGS NWIS pcode 00681", "1972–2026 月均；33,048 观测格"],
        ["站点图", "NLDI / NHDPlus", "571 站，562 条有向边（上游→下游）"],
        ["动态特征", "水温 / 流量", "与 DOC 对齐的月序列"],
        ["边物理", "NHDPlus VAA", "河段长度、汇水面积、坡度、Strahler 级"],
        ["生态 context", "EPA StreamCat", "土地利用、气候态、土壤有机质、海拔、基流"],
        ["最大连通分量", "图算法", "407 节点（E3 采样域）"],
    ]
    add_table(s, 0.7, 1.55, 12, rows, col_widths=[2.2, 3.5, 6.3])

    add_textbox(
        s, 0.7, 5.0, 12, 1.7,
        [
            ("可复现流水线（脚本化，data/ 可重新拉取）", 14, True, NAVY),
            ("fetch_doc_inventory → build_graph → build_dataset → fetch_streamcat → generate_masks → run_freeze", 12, False, GRAY),
            ("泄漏规则：模型只在 train 上拟合；测试格的 DOC 通道始终置零；StreamCat 为静态汇水区属性，不含评测期信息", 13, False, INK),
        ],
    )
    add_footer(s, 7)
    notes(s, "强调数据全部公开、脚本可复现。泄漏规则是审稿人会盯的点，主动说清楚。")

    # ── Protocol ─────────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "评测协议", "同一 masks 冻结，所有模型公平对比")
    rows = [
        ["实验", "设定", "回答的问题"],
        ["E1", "随机遮罩 20/40/60% × seed 42/43/44", "拓扑是否有用？重缺失下能恢复多少？"],
        ["E2a", "训 ≤2020，测 2021+ 全部观测", "纯特征驱动的未来外推"],
        ["E2b", "同上，但未来 20% 观测作 context", "监测网运行中的缺口填补（主时间实验）"],
        ["E3", "留出 20% 站点（其全部月份）× 3 seeds", "能否沿河网向未监测站传播信息"],
    ]
    add_table(s, 0.7, 1.55, 12, rows, col_widths=[1.2, 5.2, 5.6])

    add_textbox(
        s, 0.7, 4.3, 12, 2.4,
        [
            ("基线", 15, True, NAVY),
            ("B0 站均值  ·  B1 欧氏 Kriging  ·  B2 Random Forest  ·  B3 MLP", 14, False, INK),
            ("", 6, False, GRAY),
            ("指标", 15, True, NAVY),
            ("RMSE / MAE / R²（mg/L 空间）+ log 空间变体；DOC 长尾，两者都报", 14, False, INK),
            ("", 6, False, GRAY),
            ("Phase 0：每个 (模型 × mask) 只训一次，逐格预测落盘 parquet，下游零重训", 14, True, GREEN),
        ],
    )
    add_footer(s, 8)
    notes(s, "四组实验对应三个RQ。强调冻结协议：预测落盘，分析不重训。")

    # ── Method ladder ────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "方法演化阶梯", "每一级只加入一个物理/生态假设，同协议消融")
    stages = [
        ("G0", "普通 GCN", "河网作无向边", "拓扑 > 随机 > 无图"),
        ("H1", "有向关系 GCN", "上/下游分权重", "时间重构首次超过 RF"),
        ("H2", "传输门控 GCN", "边物理 gate", "E2a/E2b 最佳"),
        ("H2E", "+ 生态 context", "StreamCat 拼接", "E3 R² 0.23→0.32"),
        ("H2X", "+ context encoder", "MLP 嵌入身份", "总体最佳"),
    ]
    # equal-size horizontal ladder (most reliable for academic projection)
    for i, (id_, name, adds, finding) in enumerate(stages):
        x = 0.45 + i * 2.52
        y = 2.15
        h = 2.55
        col = GREEN if id_ == "H2X" else NAVY
        fill = RGBColor(0xEE, 0xF6, 0xF1) if id_ == "H2X" else WHITE
        add_rect(s, x, y, 2.35, h, fill, col)
        badge = add_rect(s, x + 0.14, y + 0.18, 0.78, 0.4, col)
        add_textbox(s, x + 0.2, y + 0.2, 0.72, 0.36, [(id_, 14, True, WHITE)])
        add_textbox(s, x + 0.14, y + 0.72, 2.05, 0.45, [(name, 13, True, col)])
        add_textbox(s, x + 0.14, y + 1.25, 2.05, 0.55, [(adds, 12, False, INK)])
        add_textbox(s, x + 0.14, y + 1.9, 2.05, 0.5, [(finding, 12, True, GRAY)])
        if i < len(stages) - 1:
            add_arrow(s, x + 2.38, y + h / 2 - 0.1, 0.12, 0.2)

    # progressive underline: darker toward H2X
    add_textbox(
        s, 0.55, 5.0, 12.2, 0.4,
        [("复杂度与信息源 →", 12, False, GRAY)],
    )
    bar_y = 5.45
    for i in range(5):
        seg = s.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0.55 + i * 2.48),
            Inches(bar_y),
            Inches(2.3),
            Inches(0.12),
        )
        # interpolate navy → green
        t = i / 4
        r = int(0x1B + (0x2C - 0x1B) * t)
        g = int(0x3A + (0x6E - 0x3A) * t)
        b = int(0x5C + (0x49 - 0x5C) * t)
        seg.fill.solid()
        seg.fill.fore_color.rgb = RGBColor(r, g, b)
        seg.line.fill.background()
        seg.shadow.inherit = False

    add_textbox(
        s, 0.55, 6.35, 12.2, 0.4,
        [("主线结论：方向优势在 E1/E2（已知站的时间重构）；生态身份优势在 E3（未见站的空间迁移）", 13, True, GREEN)],
    )
    add_footer(s, 9)
    notes(s, "这是整场汇报的骨架。每级命名+加了什么+发现了什么。最后一句是主线结论。")

    add_figure_slide(
        prs,
        "Figure 4 · 模型迭代：从通用 GNN 到河网感知编码",
        "fig4_model_iteration.png",
        "G0 连通 → H1 显式流向 → H1.5 有向稳定化 → H2 边特征门控；设计原则：逐步注入河流过程知识。",
        "Fig4讲编码如何变「河网感知」：G0只知道连谁；H1知道上下游；H1.5稳定训练；H2用边物理加权消息。注意本图止于H2，后面H2E/H2X再加生态身份。",
        10,
    )

    # ── Mechanism ────────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "关键机制：从邻域平均到输运 + 身份")
    # left: message passing sketch
    add_textbox(s, 0.7, 1.45, 5.8, 0.35, [("有向消息传递 + 边门控", 15, True, NAVY)])
    # nodes
    nodes = [(1.0, 2.2, "上游"), (3.2, 2.2, "本站"), (3.2, 4.0, "下游")]
    for x, y, lab in nodes:
        c = add_rect(s, x, y, 1.4, 0.7, WHITE, LINE)
        add_textbox(s, x + 0.15, y + 0.15, 1.1, 0.4, [(lab, 13, True, INK)])
    # arrows
    a1 = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(2.45), Inches(2.4), Inches(0.7), Inches(0.22))
    a1.fill.solid(); a1.fill.fore_color.rgb = NAVY; a1.line.fill.background(); a1.shadow.inherit = False
    a2 = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(3.7), Inches(2.95), Inches(0.22), Inches(0.9))
    a2.fill.solid(); a2.fill.fore_color.rgb = GRAY; a2.line.fill.background(); a2.shadow.inherit = False

    add_textbox(
        s, 0.7, 4.95, 5.8, 1.7,
        [
            ("· W_up / W_down 分离：上游是因果输入，下游是反向约束", 12, False, INK),
            ("· gate(e)=MLP(河段长, 汇水面积, 坡度, 河级…)", 12, False, INK),
            ("· 源头站（无上游）由节点自身 regime 表达", 12, False, INK),
        ],
    )

    # right: ecological context
    add_textbox(s, 7.0, 1.45, 5.6, 0.35, [("生态 context encoder（H2E / H2X）", 15, True, NAVY)])
    blocks = [
        ("水文", "汇水面积 / 海拔 / 坡度 / 河级"),
        ("气候", "降水 / 温度气候态"),
        ("土地利用", "林 / 农 / 城 / 湿地 %"),
        ("土壤", "土壤有机碳"),
    ]
    for i, (t, d) in enumerate(blocks):
        y = 1.95 + i * 0.85
        add_rect(s, 7.0, y, 1.5, 0.65, FILL)
        add_textbox(s, 7.1, y + 0.12, 1.3, 0.4, [(t, 12, True, NAVY)])
        add_textbox(s, 8.7, y + 0.12, 3.8, 0.5, [(d, 12, False, INK)])

    add_textbox(
        s, 7.0, 5.5, 5.6, 1.1,
        [
            ("H2E：特征直接拼接", 12, False, INK),
            ("H2X：MLP 编码后作为节点身份嵌入", 12, True, GREEN),
            ("拓扑负责传播，身份负责「这是什么河」", 12, True, GREEN),
        ],
    )
    add_footer(s, 11)
    notes(s, "左边讲方向和边门控，右边讲生态身份。点出源头站问题是H1崩E3的诊断入口。")

    # ── Results table ────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "主结果（MAE, mg/L，越低越好）", "E1 为 3 seeds 均值；† 部分 H2X 格子待全量冻结补全")
    rows = [
        ["模型", "E1 r20", "E1 r40", "E1 r60", "E2a", "E2b", "E3"],
        ["站均值", "1.46", "1.48", "1.48", "1.14", "1.12", "2.59"],
        ["Kriging", "2.15", "2.18", "2.22", "n/a", "1.51", "2.05"],
        ["Random Forest", "1.42", "1.44", "1.46", "1.12", "1.11", "2.07"],
        ["MLP", "1.87", "1.86", "1.88", "1.31", "1.29", "2.02"],
        ["G0 river GCN", "1.74", "1.77", "1.87", "1.45", "1.38", "1.97"],
        ["H1 directed", "1.50", "1.55", "1.62", "1.09", "1.07", "2.03"],
        ["H2 transport", "1.41", "1.43", "1.52", "1.10", "1.05", "2.01"],
        ["H2E + ecology", "1.41", "1.42", "1.50", "1.18", "1.13", "1.81"],
        ["H2X (ours)", "1.40", "†", "†", "†", "0.96", "1.79"],
    ]
    add_table(s, 0.55, 1.45, 12.2, rows, col_widths=[2.6, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])

    add_textbox(
        s, 0.55, 5.85, 12.2, 1.0,
        [
            ("读表：E1 上与 RF 接近并略优；E2b 时间重构 MAE 0.96 明显领先；E3 空间迁移 1.79 优于全部基线与前序 GNN", 13, False, INK),
            ("权威表以 experiments/frozen_results/benchmark.csv 为准（含 log 空间 R²）", 11, False, GRAY),
        ],
    )
    add_footer(s, 12)
    notes(s, "主表。先读基线，再读阶梯。突出H2X在E2b和E3。说明部分格子等冻结。")

    add_figure_slide(
        prs,
        "Figure 3 · 关键结果总览",
        "fig3_key_results.png",
        "任务难度（E3 最难）、E3 按 seed 对比、方法平均 R² 与 take-home 信息。",
        "对应Fig3：E3是空间最难设定；拓扑GNN有竞争力但空间外推仍难；与后文H2E/H2X修复形成递进。",
        13,
    )

    # ── E3 diagnosis ─────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "E3 诊断：错误集中在源头站", "H1 崩溃不是容量问题，而是「无上游信息可传」")
    # three finding cards
    findings = [
        ("现象", "H1 在 E3 seed42 R² 一度为负\n（clamp 修复后 ≈0.27）", NAVY),
        ("定位", "误差集中在 HUC2 10/11\nupstream_degree=0 的源头站", NAVY),
        ("修复", "H2E/H2X 引入生态身份后\n源头站 MAE 2.79 → 2.21", GREEN),
    ]
    for i, (t, d, col) in enumerate(findings):
        x = 0.7 + i * 4.15
        add_rect(s, x, 1.55, 3.85, 2.0, WHITE, LINE)
        add_textbox(s, x + 0.2, 1.75, 3.4, 0.4, [(t, 14, True, col)])
        add_textbox(s, x + 0.2, 2.3, 3.4, 1.0, [(d, 13, False, INK)])

    add_textbox(
        s, 0.7, 3.85, 12, 2.8,
        [
            ("科学含义", 16, True, NAVY),
            ("· 有向消息传递默认「每个节点都有信息量上游」——河网并不满足", 14, False, INK),
            ("· 两个图角色相同的源头站（高山林地 vs 农业坡地）DOC 制度可以完全不同", 14, False, INK),
            ("· 因此空间外推必须同时回答「连到谁」和「自己是什么」", 14, False, INK),
            ("· 这是 H2E→H2X 设计的直接动机，也是论文的可辩护故事线", 14, True, GREEN),
        ],
    )
    add_footer(s, 14)
    notes(s, "这一页是方法演化的「为什么」。诊断驱动设计，而不是堆模块。")

    # ── Extreme case ─────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "极端案例：06438000", "系统性欠预测在科学上可解释，而非单纯拟合失败")
    stats = [
        ("28", "个月有 DOC 观测\n（记录极稀疏）"),
        ("460", "峰值 mg/L\n（1980-07）"),
        ("1978–80", "高值集中期\n均值 137.5"),
    ]
    for i, (n, lab) in enumerate(stats):
        x = 0.7 + i * 4.15
        add_rect(s, x, 1.55, 3.85, 1.7, WHITE, LINE)
        add_textbox(s, x + 0.2, 1.7, 3.4, 0.7, [(n, 28, True, NAVY)])
        add_textbox(s, x + 0.2, 2.5, 3.4, 0.6, [(lab, 12, False, GRAY)])

    add_textbox(
        s, 0.7, 3.55, 12, 3.1,
        [
            ("标签审计要点", 15, True, NAVY),
            ("· 高 DOC 与流量几乎无关（Spearman≈0），不是同月洪峰冲刷", 13, False, INK),
            ("· 上游邻站始终低 DOC → 高值是局部汇水区/时期现象，非干流传播", 13, False, INK),
            ("· 1981 年后同站再未复现 → 更像 regime / 采样年代效应", 13, False, INK),
            ("", 6, False, GRAY),
            ("对模型的含义", 15, True, NAVY),
            ("月均流量 + 空间 GNN 缺少事件与源历史状态，几乎不可能凭空输出 460；", 13, False, INK),
            ("论文表述应为：与历史局部高 DOC 制度一致的、可解释的欠预测。", 13, True, GREEN),
        ],
    )
    add_footer(s, 15)
    notes(s, "用极端站说明：不是所有误差都是模型失败。这是审稿防御点。")

    # ── Limitations ──────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "当前局限", "阶段性诚实清单")
    items = [
        ("冻结未完成", "全量 parquet 仍在跑；H2X 部分 E1 格子待补齐后再出终表与四图"),
        ("时序模块未上", "H3（GRU 时序记忆）在 design 中规划但尚未进入主结果"),
        ("极端 regime", "1970s 局部高 DOC 无法由月均特征驱动复现；需事件尺度或源历史"),
        ("单流域验证", "结论目前只在密西西比；跨流域迁移与可移植性未测"),
        ("图与标签限制", "监测站非均匀空间采样；部分支流观测极稀；最大连通分量 407 节点"),
        ("消融深度", "H2X 生态组分 ablation 有初步 JSON，尚未全部进入冻结主表"),
    ]
    for i, (t, d) in enumerate(items):
        col = i % 2
        row = i // 2
        x = 0.7 + col * 6.2
        y = 1.55 + row * 1.55
        add_rect(s, x, y, 5.9, 1.35, WHITE, LINE)
        add_textbox(s, x + 0.2, y + 0.15, 5.5, 0.35, [(t, 14, True, RED)])
        add_textbox(s, x + 0.2, y + 0.55, 5.5, 0.65, [(d, 12, False, INK)])

    add_footer(s, 16)
    notes(s, "主动列局限，显得可控。冻结、时序、外流域是后续优先级。")

    # ── Future ───────────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "后续工作安排", "先交付可发表包，再扩展方法与外推")
    phases = [
        ("P0 本周", "完成全量冻结\n生成 Figure 1–4\n更新权威 benchmark 表", GREEN),
        ("P1 论文图", "Fig1 方法阶梯\nFig2 E3 对比地图\nFig3 环境贡献\nFig4 极端案例", NAVY),
        ("P2 方法", "H3 时序记忆\n生态组分完整 ablation\n不确定度 / 预测区间", NAVY),
        ("P3 扩展", "第二流域验证\n跨流域迁移\n事件尺度协变量", GRAY),
    ]
    for i, (title, body, col) in enumerate(phases):
        x = 0.7 + i * 3.15
        add_rect(s, x, 1.6, 2.95, 3.6, WHITE, col)
        head = add_rect(s, x, 1.6, 2.95, 0.55, col)
        add_textbox(s, x + 0.15, 1.68, 2.65, 0.4, [(title, 14, True, WHITE)])
        lines = body.split("\n")
        add_textbox(s, x + 0.15, 2.35, 2.65, 2.6,
                    [(ln, 13, False, INK) for ln in lines])

    add_textbox(
        s, 0.7, 5.5, 12, 1.2,
        [
            ("写作路径：以「诊断驱动的消融阶梯」为主线，突出 E2b 运行监测场景 + E3 源头站修复故事", 13, False, INK),
            ("当前即可汇报的成果：方法阶梯闭合、主指标优势、E3 诊断闭环、冻结基础设施就绪", 13, True, GREEN),
        ],
    )
    add_footer(s, 17)
    notes(s, "给出时间感：先冻结出图，再H3，再外流域。避免听起来无限扩展。")

    # ── Summary ──────────────────────────────────────────────
    s = blank(prs)
    add_title_bar(s, prs, "小结")
    points = [
        ("1", "问题", "流域 DOC 观测仅 8.9% 格子可用；河网是被忽略的强先验"),
        ("2", "方法", "G0→H1→H2→H2E→H2X 消融阶梯：拓扑 → 方向 → 传输 → 生态身份"),
        ("3", "结果", "E2b MAE 0.96、E3 MAE 1.79；源头站误差可被生态 context 显著修复"),
        ("4", "状态", "主结论可汇报；全量冻结与论文四图进行中；H3 与外流域为下一步"),
    ]
    for i, (n, tag, text) in enumerate(points):
        y = 1.55 + i * 1.15
        add_rect(s, 0.7, y, 0.7, 0.7, NAVY)
        add_textbox(s, 0.85, y + 0.15, 0.5, 0.4, [(n, 18, True, WHITE)])
        add_textbox(s, 1.7, y + 0.05, 1.3, 0.4, [(tag, 16, True, NAVY)])
        add_textbox(s, 3.1, y + 0.08, 9.5, 0.55, [(text, 15, False, INK)])

    add_textbox(
        s, 0.7, 6.3, 12, 0.5,
        [("谢谢，欢迎讨论  ·  代码与 masks：D:\\river_graph", 13, False, GRAY)],
    )
    add_footer(s, 18)
    notes(s, "四句话收束。停在「可汇报的阶段成果」和「明确的下一步」。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
