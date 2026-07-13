from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "report" / "deep_frontier_ai_forecast.md"
DEFAULT_OUTPUT = ROOT / "report" / "deep_frontier_ai_forecast.pdf"
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5F6B7A")
LINE = colors.HexColor("#C9D3DF")
TABLE_HEAD = colors.HexColor("#E8EEF5")
TABLE_ALT = colors.HexColor("#F7F9FC")
ACCENT = colors.HexColor("#6941C6")


def register_fonts() -> tuple[str, str]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for regular in candidates:
        bold = regular.with_name("Arial Bold.ttf")
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            return "ReportSans", "ReportSans-Bold"
    return "Helvetica", "Helvetica-Bold"


REGULAR_FONT, BOLD_FONT = register_fonts()


def inline_markup(value: str) -> str:
    text = html.escape(value.strip())
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r'<link href="\2" color="#365F91">\1</link>', text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName=BOLD_FONT,
            fontSize=26,
            leading=31,
            alignment=TA_CENTER,
            textColor=INK,
            spaceAfter=8 * mm,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=BOLD_FONT,
            fontSize=17,
            leading=21,
            textColor=INK,
            spaceBefore=6 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName=BOLD_FONT,
            fontSize=12.5,
            leading=16,
            textColor=ACCENT,
            spaceBefore=4 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=8.8,
            leading=12.2,
            textColor=INK,
            spaceAfter=2.2 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=8.5,
            leading=11.5,
            leftIndent=4 * mm,
            textColor=INK,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=7.2,
            leading=9,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=base["BodyText"],
            fontName=REGULAR_FONT,
            fontSize=5.7,
            leading=7.1,
            textColor=INK,
        ),
        "cell_head": ParagraphStyle(
            "CellHead",
            parent=base["BodyText"],
            fontName=BOLD_FONT,
            fontSize=5.7,
            leading=7.1,
            textColor=INK,
        ),
    }


def page_chrome(canvas, doc) -> None:
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.35)
    canvas.line(10 * mm, 10 * mm, width - 10 * mm, 10 * mm)
    canvas.setFont(REGULAR_FONT, 6.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(10 * mm, 6 * mm, "Deep Frontier AI Analysis - reproducible statistical report")
    canvas.drawRightString(width - 10 * mm, 6 * mm, f"Page {doc.page}")
    canvas.restoreState()


def markdown_table(lines: list[str], style_map: dict[str, ParagraphStyle], available_width: float) -> Table | None:
    parsed = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(parsed) < 2:
        return None
    rows = [parsed[0]] + [row for row in parsed[2:] if row]
    if not rows or not rows[0]:
        return None
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    data = []
    for row_index, row in enumerate(normalized):
        cell_style = style_map["cell_head"] if row_index == 0 else style_map["cell"]
        data.append([Paragraph(inline_markup(cell), cell_style) for cell in row])
    col_widths = [available_width / columns] * columns
    table = Table(data, colWidths=col_widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.2),
        ("TOPPADDING", (0, 0), (-1, -1), 2.3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3),
    ]
    for row_index in range(2, len(data), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), TABLE_ALT))
    table.setStyle(TableStyle(commands))
    return table


def markdown_image(line: str, source: Path, style_map: dict[str, ParagraphStyle], available_width: float):
    match = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", line.strip())
    if not match:
        return None
    caption, target = match.groups()
    path = (source.parent / target).resolve()
    if not path.exists():
        return Paragraph(f"Missing figure: {html.escape(target)}", style_map["caption"])
    image = Image(str(path))
    max_width = available_width
    max_height = 145 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return KeepTogether([image, Paragraph(inline_markup(caption), style_map["caption"])])


def build_story(source: Path, available_width: float) -> list:
    style_map = styles()
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline_markup(" ".join(paragraph)), style_map["body"]))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            table = markdown_table(table_lines, style_map, available_width)
            if table:
                story.extend([table, Spacer(1, 4 * mm)])
            continue
        image = markdown_image(stripped, source, style_map, available_width)
        if image is not None:
            flush_paragraph()
            story.append(image)
            index += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[2:]), style_map["title"]))
        elif stripped.startswith("## "):
            flush_paragraph()
            if story:
                story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(inline_markup(stripped[3:]), style_map["h2"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(stripped[4:]), style_map["h3"]))
        elif re.match(r"^(?:[-*]|\d+\.)\s+", stripped):
            flush_paragraph()
            item_text = re.sub(r"^(?:[-*]|\d+\.)\s+", "", stripped)
            bullet_type = "1" if re.match(r"^\d+\.", stripped) else "bullet"
            story.append(
                ListFlowable(
                    [ListItem(Paragraph(inline_markup(item_text), style_map["bullet"]))],
                    bulletType=bullet_type,
                    leftIndent=7 * mm,
                    bulletFontName=REGULAR_FONT,
                    bulletFontSize=7,
                    spaceAfter=1 * mm,
                )
            )
        elif stripped == "---":
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
        else:
            paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def render(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    margin_x = 10 * mm
    margin_top = 10 * mm
    margin_bottom = 13 * mm
    doc = BaseDocTemplate(
        str(output),
        pagesize=(page_width, page_height),
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_top,
        bottomMargin=margin_bottom,
        title="Deep Frontier AI Analysis",
        author="AI Capability Signals",
    )
    frame = Frame(
        margin_x,
        margin_bottom,
        page_width - 2 * margin_x,
        page_height - margin_top - margin_bottom,
        id="content",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(PageTemplate(id="analysis", frames=[frame], onPage=page_chrome))
    doc.build(build_story(source, page_width - 2 * margin_x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the generated Markdown analysis as a verified landscape PDF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
