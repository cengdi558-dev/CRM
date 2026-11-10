"""Apply a figure plan to a .docx: swap images and rewrite their captions.

    python -m crm.apply_figures --docx paper.docx --plan plan.json --out paper_v2.docx

``plan.json``::

    {
      "replace": [
        {"index": 43, "image": "figs/fig_attention.png",
         "caption": "Fig. 9. Gradient saliency of CRM under clean and occluded conditions. ..."},
        {"index": 44, "image": "figs/fig_challenging.png",
         "caption": "Fig. 10. Saliency under challenging conditions: ..."}
      ]
    }

Only the listed media parts and the caption paragraphs that start with the same
``Fig. N`` label are touched; everything else is copied byte for byte.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile

from lxml import etree

from .replace_figures import fit_to_frame, list_figures

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def caption_label(text: str) -> str | None:
    m = re.match(r"\s*((?:Fig\.?|Figure)\s*\.?\s*\d+)", text, re.I)
    return m.group(1) if m else None


def rewrite_caption(root, label: str, new_text: str) -> bool:
    """Replace the text of the paragraph whose label matches ``label``."""
    target = None
    for p in root.iter(f"{W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{W}t")).strip()
        if caption_label(text) and caption_label(text).replace(" ", "").lower() == label.replace(" ", "").lower():
            target = p
            break
    if target is None:
        return False

    runs = list(target.iter(f"{W}t"))
    if not runs:
        return False
    runs[0].text = new_text
    for run in runs[1:]:
        run.text = ""
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as fh:
        plan = json.load(fh)

    figures = list_figures(args.docx)
    replacements: dict[str, bytes] = {}
    caption_edits: list[tuple[str, str]] = []

    for item in plan["replace"]:
        idx = int(item["index"])
        if not (0 <= idx < len(figures)):
            raise SystemExit(f"index {idx} out of range")
        fig = figures[idx]
        replacements[fig["part"]] = fit_to_frame(item["image"], fig["px"])
        print(f"#{idx} {fig['part']} <- {item['image']} ({fig['px'][0]}x{fig['px'][1]})")

        if item.get("caption"):
            label = item.get("label") or f"Fig. {item.get('fig', '')}".strip()
            caption_edits.append((label, item["caption"]))

    with zipfile.ZipFile(args.docx) as src:
        doc_xml = etree.fromstring(src.read("word/document.xml"))
        for label, text in caption_edits:
            ok = rewrite_caption(doc_xml, label, text)
            print(f"caption {label}: {'updated' if ok else 'NOT FOUND'}")

        new_doc = etree.tostring(doc_xml, xml_declaration=True, encoding="UTF-8", standalone=True)

        with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if item.filename == "word/document.xml":
                    dst.writestr(item, new_doc)
                elif item.filename in replacements:
                    dst.writestr(item, replacements[item.filename])
                else:
                    dst.writestr(item, src.read(item.filename))

    print(f"written {args.out}")


if __name__ == "__main__":
    main()
