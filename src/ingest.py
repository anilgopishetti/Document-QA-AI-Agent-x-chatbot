
import os
import json
import uuid
import argparse
import logging
from typing import List, Dict, Any
import fitz                # PyMuPDF
import pdfplumber
import pandas as pd
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:200]


def extract_metadata(fitz_doc: fitz.Document) -> Dict[str, Any]:
    md = fitz_doc.metadata or {}
    return {
        "title": md.get("title", "").strip(),
        "author": md.get("author", "").strip(),
        "subject": md.get("subject", "").strip(),
        "producer": md.get("producer", "").strip(),
        "creationDate": md.get("creationDate"),
        "modDate": md.get("modDate"),
        "num_pages": fitz_doc.page_count,
    }


def extract_text_blocks(path: str) -> Dict[str, Any]:
    """
    Use PyMuPDF to extract page-level blocks with font-size/span info.
    Returns a structure: pages -> list of blocks (each block has text, bbox, max_font_size)
    """
    doc = fitz.open(path)
    pages = []
    for p in range(doc.page_count):
        page = doc[p]
        page_dict = page.get_text("dict")  # contains "blocks"
        blocks_out = []
        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:
                continue  # skip images/others; text blocks have type==0
            # accumulate text and collect font sizes
            block_text = ""
            font_sizes = []
            fonts = set()
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    block_text += txt
                    font_sizes.append(span.get("size", 0))
                    fonts.add(span.get("font", ""))
            if not block_text.strip():
                continue
            blocks_out.append({
                "text": block_text.strip(),
                "bbox": b.get("bbox"),
                "font_sizes": font_sizes,
                "max_font_size": max(font_sizes) if font_sizes else None,
                "avg_font_size": (sum(font_sizes) / len(font_sizes)) if font_sizes else None,
                "fonts": list(fonts),
            })
        pages.append({"page_number": p + 1, "blocks": blocks_out})
    doc.close()
    return {"pages": pages}


def detect_repeated_headers(pages_blocks: List[Dict]) -> List[str]:
    """
    Detect repeated short lines across pages (likely headers/footers).
    Return list of repeated strings to be filtered out.
    """
    phrase_counter = Counter()
    for page in pages_blocks:
        for block in page["blocks"]:
            text = block["text"].strip()
            # consider very short blocks likely headers/footers
            if len(text) < 120:
                phrase_counter[text] += 1
    # any phrase appearing on >30% of pages is suspicious
    threshold = max(2, int(0.3 * len(pages_blocks)))
    repeated = [p for p, c in phrase_counter.items() if c >= threshold]
    return repeated


def remove_repeated_headers_from_blocks(pages_blocks: List[Dict], repeated: List[str]):
    if not repeated:
        return pages_blocks
    for page in pages_blocks:
        filtered = []
        for b in page["blocks"]:
            if b["text"].strip() in repeated:
                continue
            filtered.append(b)
        page["blocks"] = filtered
    return pages_blocks


def detect_headings_and_sections(pages_blocks: List[Dict]) -> List[Dict]:
    """
    Heuristic:
      - Collect all span font sizes and compute a global threshold for 'heading' (e.g., 90th percentile).
      - A block with max_font_size >= threshold will be treated as a heading.
      - Build sections by splitting text when heading found.
    Returns list of sections: {"section_id","heading","text","start_page","end_page"}
    """
    all_font_sizes = []
    for page in pages_blocks:
        for b in page["blocks"]:
            if b.get("max_font_size"):
                all_font_sizes.append(b["max_font_size"])
    if not all_font_sizes:
        # fallback: no font info, create one section per page
        sections = []
        for p in pages_blocks:
            page_text = "\n\n".join([b["text"] for b in p["blocks"]])
            sections.append({
                "section_id": str(uuid.uuid4()),
                "heading": f"Page {p['page_number']}",
                "text": page_text,
                "start_page": p["page_number"],
                "end_page": p["page_number"],
            })
        return sections

    # compute a high-percentile threshold for heading sizes
    import numpy as np
    threshold = float(np.percentile(all_font_sizes, 90))
    sections = []
    current = {"heading": None, "text": "", "start_page": None, "end_page": None}

    for p in pages_blocks:
        for b in p["blocks"]:
            if b.get("max_font_size") and b["max_font_size"] >= threshold:
                # treat this block as heading: start new section
                if current["heading"] or current["text"]:
                    # finalize current
                    sections.append({
                        "section_id": str(uuid.uuid4()),
                        "heading": current["heading"] or "Untitled",
                        "text": current["text"].strip(),
                        "start_page": current["start_page"],
                        "end_page": current["end_page"],
                    })
                current = {
                    "heading": b["text"].strip(),
                    "text": "",
                    "start_page": p["page_number"],
                    "end_page": p["page_number"],
                }
            else:
                # append to current section
                if current["start_page"] is None:
                    current["start_page"] = p["page_number"]
                current["end_page"] = p["page_number"]
                current["text"] += ("\n\n" + b["text"].strip())
    # push last
    if current["heading"] or current["text"]:
        sections.append({
            "section_id": str(uuid.uuid4()),
            "heading": current["heading"] or "Introduction",
            "text": current["text"].strip(),
            "start_page": current["start_page"],
            "end_page": current["end_page"],
        })
    return sections


def extract_tables(path: str, output_dir: str) -> List[Dict]:
    """
    Use pdfplumber to extract tables. Save each table as CSV and return metadata.
    """
    tables_meta = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                page_tables = page.extract_tables()
            except Exception as e:
                logger.warning(f"pdfplumber failed on page {i} of {path}: {e}")
                page_tables = []
            for tidx, table in enumerate(page_tables, start=1):
                # table is list-of-lists
                df = pd.DataFrame(table)
                fname = f"{sanitize_filename(Path(path).stem)}_p{i}_table{tidx}.csv"
                csv_path = os.path.join(output_dir, "tables", fname)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False, header=False)
                tables_meta.append({
                    "table_id": str(uuid.uuid4()),
                    "page": i,
                    "csv_path": csv_path,
                    "rows": len(table),
                    "cols": len(table[0]) if table and len(table) > 0 else 0,
                    "raw": table  # careful with large tables; optional
                })
    return tables_meta


def extract_images(path: str, output_dir: str) -> List[Dict]:
    """
    Extract embedded images using PyMuPDF.
    """
    images_meta = []
    doc = fitz.open(path)
    for pageno in range(doc.page_count):
        page = doc[pageno]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            fname = f"{sanitize_filename(Path(path).stem)}_p{pageno+1}_img{img_index}.{ext}"
            out_dir = os.path.join(output_dir, "images")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            images_meta.append({
                "image_id": str(uuid.uuid4()),
                "page": pageno + 1,
                "path": out_path,
                "ext": ext,
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            })
    doc.close()
    return images_meta


def find_references(pages_blocks: List[Dict]) -> str:
    """
    Scan for blocks whose text is 'References' (case-insensitive) and gather subsequent blocks/pages.
    """
    found = False
    ref_texts = []
    for page in pages_blocks:
        for b in page["blocks"]:
            t = b["text"].strip()
            if not found and t.lower() in ("references", "reference", "bibliography", "works cited"):
                found = True
                continue
            if found:
                ref_texts.append(t)
        if found and page["blocks"] == []:
            # in case references start on a new page with image only
            continue
    return "\n\n".join(ref_texts).strip()


def needs_ocr_check(path: str, page_sample_count: int = 3) -> bool:
    """
    Basic check: if many pages have very little extracted text -> probably scanned (image) PDF.
    """
    doc = fitz.open(path)
    empty_pages = 0
    sample = min(page_sample_count, doc.page_count)
    for i in range(sample):
        text = doc[i].get_text("text").strip()
        if len(text) < 50:
            empty_pages += 1
    doc.close()
    return empty_pages >= max(1, sample // 2)


def process_pdf(path: str, output_dir: str) -> Dict[str, Any]:
    logger.info(f"Processing {path}")
    doc = fitz.open(path)
    metadata = extract_metadata(doc)
    doc.close()

    text_structure = extract_text_blocks(path)
    pages_blocks = text_structure["pages"]

    # detect & remove repeated headers/footers
    repeated = detect_repeated_headers(pages_blocks)
    if repeated:
        pages_blocks = remove_repeated_headers_from_blocks(pages_blocks, repeated)

    # sections & headings
    sections = detect_headings_and_sections(pages_blocks)

    # tables (pdfplumber)
    tables_meta = extract_tables(path, output_dir)

    # images
    images_meta = extract_images(path, output_dir)

    # references
    references_text = find_references(pages_blocks)

    # raw text fallback
    full_text = "\n\n".join([b["text"] for p in pages_blocks for b in p["blocks"]])

    # scanned check
    scanned = needs_ocr_check(path)

    out = {
        "doc_id": str(uuid.uuid4()),
        "filename": os.path.basename(path),
        "title": metadata.get("title") or (sections[0]["heading"] if sections else Path(path).stem),
        "metadata": metadata,
        "sections": sections,
        "tables": tables_meta,
        "figures": images_meta,
        "references": references_text,
        "raw_text": full_text,
        "needs_ocr": scanned,
    }

    # save json
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{sanitize_filename(Path(path).stem)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved processed document JSON to {out_path}")
    return out


def batch_process(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    manifest = []
    for pdf in sorted(input_dir.glob("*.pdf")):
        try:
            doc_out = process_pdf(str(pdf), str(output_dir))
            manifest.append({
                "doc_id": doc_out["doc_id"],
                "filename": doc_out["filename"],
                "json_path": str(Path(output_dir) / f"{sanitize_filename(pdf.stem)}.json")
            })
        except Exception as e:
            logger.exception(f"Failed to process {pdf}: {e}")
    # write manifest
    manifest_path = output_dir / "documents_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Wrote manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch ingest PDFs and extract structured content")
    parser.add_argument("--input_dir", required=True, help="Folder containing PDFs")
    parser.add_argument("--output_dir", required=True, help="Folder to write JSON/tables/images")
    args = parser.parse_args()
    batch_process(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
