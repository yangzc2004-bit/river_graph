"""Convert the checked manuscript into a self-contained, editable LaTeX paper."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "RiverDOCNet_密西西比河DOC重构小论文.docx"
TARGET = OUT / "riverdocnet_mississippi.tex"

PREAMBLE = r"""% Compile twice with XeLaTeX (TeX Live 2026).
% All tables, equations, references and the vector diagram are embedded.
\documentclass[UTF8,11pt,a4paper,fontset=none]{ctexart}
\usepackage[margin=24mm,headheight=14pt]{geometry}
\usepackage{fontspec}
\IfFontExistsTF{Songti SC}{
  \setCJKmainfont{Songti SC}[BoldFont=Heiti SC]
  \setCJKsansfont{Heiti SC}
  \setCJKmonofont{Heiti SC}
}{
  \setCJKmainfont{FandolSong-Regular}[BoldFont=FandolSong-Bold]
  \setCJKsansfont{FandolHei-Regular}[BoldFont=FandolHei-Bold]
  \setCJKmonofont{FandolFang-Regular}
}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{\setsansfont{TeX Gyre Heros}}
\usepackage{amsmath,amssymb,booktabs,array,longtable,graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{caption,placeins,fancyhdr}
\usepackage[hidelinks,unicode]{hyperref}
\urlstyle{same}
\captionsetup{font=small,labelfont=bf,labelsep=quad}
\ctexset{
  section={format=\large\bfseries,beforeskip=1.4ex plus .3ex,afterskip=.8ex},
  subsection={format=\normalsize\bfseries,beforeskip=1.1ex,afterskip=.5ex}
}
\setlength{\parindent}{2em}
\setlength{\parskip}{.2em}
\linespread{1.18}
\setlength{\emergencystretch}{2em}
\setlength{\LTpre}{.6em}
\setlength{\LTpost}{.4em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.18}
\widowpenalty=10000
\clubpenalty=10000
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small RiverDOCNet：密西西比河 DOC 时空重构}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{.3pt}
\renewcommand{\footrulewidth}{0pt}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\title{\bfseries RiverDOCNet\\[.35em]密西西比河流域溶解性有机碳时空重构}
\author{}
\date{}
\begin{document}
\maketitle
\thispagestyle{plain}
\vspace{-2.4em}
"""

EQUATIONS = [
    r"\widehat y_{i,t}=f_\theta\!\left(\mathbf e_i,\mathbf w_{i,t-11:t},G\right)",
    r"\mathbf u_i=W_2\,\operatorname{GELU}(W_1\mathbf e_i+b_1)+b_2",
    (r"\mathbf g_{i,t}=\operatorname{GRU}(\mathbf w_{i,t-11:t}),\qquad "
     r"\mathbf h^{(0)}_{i,t}=W_f[\mathbf u_i\Vert\mathbf g_{i,t}\Vert\mathbf s_t]+b_f"),
    (r"\alpha^{(h)}_{ij,t}=\underset{j\in\mathcal N_i^{(h)}}{\operatorname{softmax}}"
     r"\!\left(\frac{(\mathbf q^{(h)}_{i,t})^\top\mathbf k^{(h)}_{j,t}}{\sqrt{d_h}}"
     r"+b_h(r_{ij})\right)"),
    r"\widehat y_{i,t}=\max\!\left\{0,\exp\!\left[\min(15,\mu+\sigma z_{i,t})\right]-1\right\}",
    (r"\mathcal L=\frac{1}{|\Omega_{\mathrm{train}}|}\sum_{(i,t)\in\Omega_{\mathrm{train}}}"
     r"\left[\frac{\log(1+y_{i,t})-\mu}{\sigma}-z_{i,t}\right]^2"),
    (r"\operatorname{MAE}=\frac1n\sum_{k=1}^n|\widehat y_k-y_k|,\qquad "
     r"R^2=1-\frac{\sum_k(y_k-\widehat y_k)^2}{\sum_k(y_k-\overline y)^2}"),
]

FIGURE = r"""\begin{figure}[!htbp]
\centering
\begin{tikzpicture}[
  x=1cm,y=1cm,
  box/.style={draw=black!45,rounded corners=2pt,align=center,
    font=\small,minimum height=1.05cm,inner sep=5pt},
  flow/.style={-{Stealth[length=2mm]},thick,draw=black!65}
]
\node[box,fill=green!8,text width=2.4cm] (env) at (0,1.7)
  {环境基底\\15 项属性};
\node[box,fill=green!8,text width=2.3cm] (mlp) at (3.4,1.7)
  {环境 MLP\\24 维表示};
\node[box,fill=blue!7,text width=2.4cm] (hydro) at (0,-.4)
  {水文时间序列\\12 个月\\8 个输入通道};
\node[box,fill=blue!7,text width=2.3cm] (gru) at (3.4,-.4)
  {时间 GRU\\12 维记忆};
\node[box,fill=orange!9,text width=2.2cm,minimum height=1.5cm] (fusion) at (6.7,.65)
  {拼接与投影\\当前季节\\24 维表示};
\node[box,fill=teal!8,text width=3.1cm,minimum height=1.6cm] (graph) at (10.8,.65)
  {两层方向分头\\图 Transformer\\上游、下游、全局头};
\node[box,fill=violet!8,text width=3.1cm] (network) at (10.8,3)
  {有向河网\\连接关系与方向约束};
\node[box,fill=orange!8,text width=3.1cm] (output) at (10.8,-1.9)
  {浓度读出与逆变换\\DOC 站点 $\times$ 月份矩阵};
\draw[flow] (env)--(mlp);
\draw[flow] (hydro)--(gru);
\draw[flow] (mlp.east)--++(.35,0)|-(fusion.155);
\draw[flow] (gru.east)--++(.35,0)|-(fusion.205);
\draw[flow] (fusion)--(graph);
\draw[flow] (network)--(graph);
\draw[flow] (graph)--(output);
\end{tikzpicture}
\caption{RiverDOCNet 方向分头方案的信息流。DOC 仅用于训练监督；预测场由具备输入的站点和月份构成。}
\label{fig:architecture}
\end{figure}
"""

KEYS = ["casas2020", "sterle2024", "lu2026", "wampler2024", "futter2007",
        "budamala2026", "cho2014", "ying2021", "hill2016"]


def escape(text: str, *, citations: bool = True) -> str:
    text = text.replace("——", "---").replace("—", "--").replace("–", "--").replace("−", "-")
    text = text.replace("～", " 至 ")
    # Escape literal source content before introducing LaTeX commands.
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}"}
    text = "".join(mapping.get(char, char) for char in text)
    text = text.replace("R²", "$R^2$").replace("↑", "$\\uparrow$").replace("↓", "$\\downarrow$")
    text = text.replace("×", "$\\times$")
    if citations:
        text = re.sub(r"\[([1-9])\]", lambda m: r"\cite{"+KEYS[int(m.group(1))-1]+"}", text)
    return text


def table_latex(table: Table, caption: str, number: int) -> str:
    counts = len(table.columns)
    if number == 1:
        widths = [4.1, 11.1]
    elif counts == 4:
        widths = [7.3, 2.5, 2.5, 2.5]
    elif counts == 6:
        widths = [6.0, 1.65, 2.0, 1.75, 1.65, 2.0]
    else:
        widths = [7.8, 1.9, 2.1, 2.0, 1.8]
    # Includes inter-column padding: maximum tabular width < 162 mm.
    if number != 1:
        scale = (14.8 if counts == 6 else 15.0) / sum(widths)
        widths = [width * scale for width in widths]
    cols = "".join(("L" if i == 0 or number == 1 else "C")+"{"+f"{width:.3f}cm"+"}"
                   for i, width in enumerate(widths))
    all_rows = []
    for row_index, row in enumerate(table.rows):
        cells = []
        for cell in row.cells:
            value = escape(cell.text)
            bold = row_index == 0 or any(run.bold for p in cell.paragraphs for run in p.runs)
            cells.append(r"\textbf{"+value+"}" if bold else value)
        all_rows.append(" & ".join(cells)+r" \\")
    title = re.sub(r"^表\s*\d+\s*", "", caption)
    header = all_rows[0]
    if number == 5:
        return "\n".join([
            r"\begingroup\centering\small", r"\setlength{\tabcolsep}{3pt}",
            r"\captionsetup{hypcap=false}",
            r"\captionof{table}{"+escape(title)+r"}\label{tab:5}",
            r"\begin{tabular}{"+cols+"}", r"\toprule", header, r"\midrule",
            *all_rows[1:], r"\bottomrule", r"\end{tabular}", r"\par\endgroup",
        ])
    result = [r"\begingroup", r"\small", r"\setlength{\tabcolsep}{3pt}",
              r"\begin{longtable}{"+cols+"}",
              r"\caption{"+escape(title)+r"}\label{tab:"+str(number)+r"}\\",
              r"\toprule", header, r"\midrule", r"\endfirsthead",
              r"\multicolumn{"+str(counts)+r"}{c}{\small 表 \thetable（续）}\\",
              r"\toprule", header, r"\midrule", r"\endhead",
              r"\midrule", r"\multicolumn{"+str(counts)+r"}{r}{\small 下页续}\\",
              r"\endfoot", r"\bottomrule", r"\endlastfoot", *all_rows[1:],
              r"\end{longtable}", r"\endgroup"]
    return "\n".join(result)


def build() -> None:
    doc = Document(SOURCE)
    parts = [PREAMBLE]
    equations, tables, references = 0, 0, []
    pending_caption = None
    in_references = False
    paragraphs = 0
    for element in doc.element.body:
        if element.tag == qn("w:tbl"):
            tables += 1
            parts.append(table_latex(Table(element, doc), pending_caption, tables))
            if tables == 5:
                parts.append(r"\end{minipage}\par\medskip")
            pending_caption = None
            continue
        if element.tag != qn("w:p"):
            continue
        p = Paragraph(element, doc)
        if element.xpath(".//m:oMath"):
            parts.append("\\begin{equation}\n"+EQUATIONS[equations]+"\n\\end{equation}")
            equations += 1
            continue
        if element.xpath(".//w:drawing"):
            parts.append(FIGURE)
            continue
        text = p.text.strip()
        text = text.replace(
            "模型和比较方案由训练及验证数据选择，不根据测试结果重新选择。",
            "每次拟合的检查点仅根据训练和验证数据确定；方向编码候选按预定的验证指标联合规则选出。"
            "本文对同一开发队列进行了多轮方法探索，因此结果仍属于流域内开发证据。",
        )
        text = text.replace(
            "土地覆盖与气候常年值用作静态背景，不被解释为每个历史月份的真实土地利用变化。",
            "2019 年土地覆盖与 1991-2020 年气候常年值用作静态背景，"
            "不被解释为每个历史月份的真实土地利用或气候变化。",
        )
        if not text or p.style.name == "Title" or text.startswith("图 1 "):
            continue
        if text == "参考文献":
            in_references = True
            continue
        if in_references:
            references.append(text)
            continue
        if text == "摘要":
            parts.append(r"\section*{摘要}")
            continue
        if text.startswith("关键词"):
            parts.append(r"\noindent\textbf{关键词：}"+escape(text.removeprefix("关键词").strip())+"\n")
            continue
        if p.style.name.startswith("Heading"):
            number = int(p.style.name.split()[-1])
            command = "section" if number == 1 else "subsection"
            title = re.sub(r"^\d+(?:\.\d+)*\s+", "", text)
            if text.startswith("5.4 "):
                parts.append(r"\par\noindent\begin{minipage}{\linewidth}")
            parts.append("\\"+command+"{"+escape(title)+"}")
            continue
        if re.match(r"^表\s*\d+\s", text):
            pending_caption = text
            continue
        parts.append(escape(text)+"\n")
        paragraphs += 1
    parts.append(r"\FloatBarrier")
    parts.append(r"\clearpage")
    parts.append(r"\begin{thebibliography}{9}")
    for key, reference in zip(KEYS, references, strict=True):
        match = re.search(r"https://\S+", reference)
        link = match.group(0)
        text = reference[:match.start()].strip()
        text = re.sub(r"^\[\d+\]\s*", "", text)
        parts.append(r"\bibitem{"+key+"} "+escape(text, citations=False)+"\n"+r"\par\nobreak\url{"+link+"}.")
    parts.extend([r"\end{thebibliography}", r"\end{document}"])
    manuscript = "\n\n".join(parts)+"\n"
    assert equations == 7 and tables == 5 and len(references) == 9
    assert not any(token in manuscript for token in
                   ("fixed_fusion", "direction_heads", "seed42", "permuted_river", "H2X", "phase4_transfer"))
    TARGET.write_text(manuscript)
    audit = {"source": SOURCE.name, "latex": TARGET.name, "body_paragraphs": paragraphs,
             "equations": equations, "tables": tables, "references": len(references),
             "embedded_vector_diagrams": 1, "external_document_assets": 0}
    (OUT/"latex_content_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    build()
