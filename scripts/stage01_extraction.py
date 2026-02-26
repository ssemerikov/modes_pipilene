#!/usr/bin/env python3
"""
Stage 1: Text Extraction & Preprocessing
Extracts text from all 4 PDFs with layout-aware methods.
"""
import re
import json
import fitz  # pymupdf
from pathlib import Path
from utils import (
    PDF_FILES, BOOK_META, OUTPUT_DIR, TEXTS_DIR,
    dehyphenate, clean_ocr_text, remove_headers_footers, save_json
)

def extract_applebaum():
    """Portrait 1-col, good OCR. Chapter segmentation from known TOC."""
    print("Extracting Applebaum...")
    doc = fitz.open(str(PDF_FILES["applebaum"]))

    # Known chapter structure from TOC (pages 17-18)
    # Page numbers in PDF are 0-indexed, content starts around p19
    chapters_spec = [
        ("Introduction to the 2015 Edition", 19),
        ("Introduction", 25),
        ("Prelude", 31),
        # Part One: Germans
        ("Kaliningrad/Königsberg", 49),
        # Part Two: Poles and Lithuanians
        ("Vilnius/Wilno", 87),
        ("Paberžė", 100),
        ("Perloja", 110),
        ("Eišiškės", 128),
        ("Radun", 134),
        ("Hermaniszki", 154),
        ("Bieniakonie", 202),
        ("Nowogródek", 213),
        # Part Three: Russians, Belarusians, and Ukrainians
        ("Minsk", 218),
        ("Brest", 222),
        ("Kobrin", 227),
        ("A Memory", 249),
        ("L'viv/Lvov/Lwów", 257),
        ("Woroniaki", 268),
        ("Drohobych", 274),
        ("Across the Carpathians", 281),
        # Part Four: Island Cities
        ("Chernivtsi/Czernowitz", 299),
        ("Kamenets Podolsky", 308),
        ("Kishinev/Chișinău", 317),
        ("Odessa", 330),
        ("Epilogue", 339),
    ]

    chapters = []
    for i, (name, start_page) in enumerate(chapters_spec):
        end_page = chapters_spec[i + 1][1] if i + 1 < len(chapters_spec) else len(doc)
        text = ""
        for p in range(start_page, min(end_page, len(doc))):
            page_text = doc[p].get_text("text")
            page_text = remove_headers_footers(page_text, p)
            text += page_text + "\n"
        text = dehyphenate(text)
        text = clean_ocr_text(text)
        if text.strip():
            chapters.append({"chapter": name, "text": text.strip()})

    doc.close()
    return chapters


def extract_nicolay():
    """Portrait 1-col, excellent OCR. Chapter segmentation from known TOC."""
    print("Extracting Nicolay...")
    doc = fitz.open(str(PDF_FILES["nicolay"]))

    # From TOC (page 9-10)
    chapters_spec = [
        ("Introduction", 15),
        # Part I
        ("The Humorless Ladies of Border Control (Ukraine)", 25),
        ("Party for Everybody (Rostov-on-Don to Saint Petersburg)", 48),
        ("A Real Lenin of Our Time (Moscow)", 74),
        ("God-Forget-It House (Trans-Siberian)", 84),
        ("The Knout and the Pierogi (Tomsk to Baikal)", 104),
        ("The Hall of Sufficient Looking (Trans-Mongolian)", 157),
        # Part II
        ("Drunk Nihilists Make a Good Audience (Croatia, Slovenia, Serbia)", 189),
        ("A Fur Coat with Morsels (Hungary, Poland)", 234),
        ("Poor, but They Have Style (Romania)", 247),
        ("You Are an Asshole Big Time (Bulgaria)", 262),
        ("Don't Bring Your Beer in Church (Bucharest to Vienna)", 296),
        # Part III
        ("Changing the Country, We Apologize (Ukraine After the Flood)", 309),
    ]

    chapters = []
    for i, (name, start_page) in enumerate(chapters_spec):
        end_page = chapters_spec[i + 1][1] if i + 1 < len(chapters_spec) else len(doc)
        text = ""
        for p in range(start_page, min(end_page, len(doc))):
            page_text = doc[p].get_text("text")
            page_text = remove_headers_footers(page_text, p)
            text += page_text + "\n"
        text = dehyphenate(text)
        text = clean_ocr_text(text)
        if text.strip():
            chapters.append({"chapter": name, "text": text.strip()})

    doc.close()
    return chapters


def extract_brumme():
    """Landscape 2-col, poor OCR. Use pymupdf blocks with column sorting."""
    print("Extracting Brumme...")
    doc = fitz.open(str(PDF_FILES["brumme"]))
    page_width = doc[0].rect.width  # ~780

    # Brumme is a diary — segment by date patterns or treat as continuous chapters
    # Content starts at page 4 (Vorwort), main text ~page 5 onward
    # Two columns: left ~30-370, right ~410-750
    midpoint = page_width / 2  # ~390

    all_text = []
    current_section = "Vorwort"
    section_text = ""

    for page_num in range(4, len(doc) - 1):  # skip cover/title pages and last blank
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        # Separate into left and right columns
        left_blocks = []
        right_blocks = []

        for b in blocks:
            if b["type"] != 0:  # skip images
                continue
            x_center = (b["bbox"][0] + b["bbox"][2]) / 2
            if x_center < midpoint:
                left_blocks.append(b)
            else:
                right_blocks.append(b)

        # Sort each column by y position
        left_blocks.sort(key=lambda b: b["bbox"][1])
        right_blocks.sort(key=lambda b: b["bbox"][1])

        # Extract text from sorted blocks
        page_text = ""
        for b in left_blocks + right_blocks:
            block_text = ""
            for line in b["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                block_text += line_text + "\n"
            page_text += block_text + "\n"

        page_text = remove_headers_footers(page_text, page_num)

        # Check for date headers that might indicate new diary entries
        date_match = re.search(
            r'^(\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|'
            r'September|Oktober|November|Dezember)\s+\d{4})',
            page_text, re.MULTILINE
        )

        if date_match and section_text.strip():
            all_text.append({"chapter": current_section, "text": section_text.strip()})
            current_section = date_match.group(1)
            section_text = page_text
        else:
            section_text += "\n" + page_text

    if section_text.strip():
        all_text.append({"chapter": current_section, "text": section_text.strip()})

    # If we only got one section, split into roughly equal parts for analysis
    if len(all_text) <= 2:
        # Combine all and split by paragraph groups
        full_text = "\n\n".join(ch["text"] for ch in all_text)
        full_text = dehyphenate(full_text)
        full_text = clean_ocr_text(full_text)
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]

        # Create roughly 6-8 sections
        chunk_size = max(1, len(paragraphs) // 7)
        all_text = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i:i + chunk_size]
            section_name = f"Section_{i // chunk_size + 1}"
            # Try to use first date found as section name
            for p in chunk:
                dm = re.search(r'(\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|'
                               r'August|September|Oktober|November|Dezember)\s+\d{4})', p)
                if dm:
                    section_name = dm.group(1)
                    break
            all_text.append({"chapter": section_name, "text": "\n\n".join(chunk)})
    else:
        for ch in all_text:
            ch["text"] = clean_ocr_text(dehyphenate(ch["text"]))

    doc.close()
    return all_text


def extract_orth():
    """Landscape 2-col, moderate OCR. Column sorting at x=400."""
    print("Extracting Orth...")
    doc = fitz.open(str(PDF_FILES["orth"]))
    page_width = doc[0].rect.width  # ~789

    # Orth has city/region chapter headers
    # Content pages start at ~page 2, photo pages at ~64+
    midpoint = page_width / 2  # ~395

    # Known chapter structure based on city names (approximate pages)
    chapters_spec = [
        ("Kyjiw – Warum diese Reise?", 2),
        ("Kyjiw – Wandel und Wandern", 12),
        ("Kyjiw – Zimmer eines Superhelden", 18),
        ("Dnipro – Ein historischer Knall", 25),
        ("Charkiw – Leben und Funktionieren", 31),
        ("Poltawa – Zwanzig Bilder pro Sekunde", 40),
        ("Karpaten", 46),
        ("Odesa", 48),
        ("Schluss – Plus oder Minus", 55),
    ]

    chapters = []
    text_end_page = 63  # After this are photo pages

    for i, (name, start_page) in enumerate(chapters_spec):
        end_page = chapters_spec[i + 1][1] if i + 1 < len(chapters_spec) else text_end_page

        section_text = ""
        for page_num in range(start_page, min(end_page, text_end_page)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            left_blocks = []
            right_blocks = []

            for b in blocks:
                if b["type"] != 0:
                    continue
                x_center = (b["bbox"][0] + b["bbox"][2]) / 2
                # Filter out sidebar info boxes (usually small, Helvetica-Bold, in margins)
                is_sidebar = False
                for line in b["lines"]:
                    for span in line["spans"]:
                        if "Helvetica-Bold" in span.get("font", "") and span["size"] < 10:
                            is_sidebar = True
                if is_sidebar and (b["bbox"][2] - b["bbox"][0]) < 200:
                    continue

                if x_center < midpoint:
                    left_blocks.append(b)
                else:
                    right_blocks.append(b)

            left_blocks.sort(key=lambda b: b["bbox"][1])
            right_blocks.sort(key=lambda b: b["bbox"][1])

            page_text = ""
            for b in left_blocks + right_blocks:
                block_text = ""
                for line in b["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    block_text += line_text + "\n"
                page_text += block_text + "\n"

            page_text = remove_headers_footers(page_text, page_num)
            section_text += page_text + "\n"

        section_text = dehyphenate(section_text)
        section_text = clean_ocr_text(section_text)
        if section_text.strip():
            chapters.append({"chapter": name, "text": section_text.strip()})

    doc.close()
    return chapters


def main():
    print("=" * 60)
    print("STAGE 1: Text Extraction & Preprocessing")
    print("=" * 60)

    corpus = {}

    extractors = {
        "applebaum": extract_applebaum,
        "nicolay": extract_nicolay,
        "brumme": extract_brumme,
        "orth": extract_orth,
    }

    for book_id, extractor in extractors.items():
        chapters = extractor()
        meta = BOOK_META[book_id]

        # Compute stats
        total_words = sum(len(ch["text"].split()) for ch in chapters)
        total_chars = sum(len(ch["text"]) for ch in chapters)

        corpus[book_id] = {
            "metadata": meta,
            "chapters": chapters,
            "stats": {
                "num_chapters": len(chapters),
                "total_words": total_words,
                "total_chars": total_chars,
            }
        }

        print(f"  {book_id}: {len(chapters)} chapters, {total_words:,} words")

        # Save individual text files
        full_text = "\n\n---\n\n".join(
            f"## {ch['chapter']}\n\n{ch['text']}" for ch in chapters
        )
        (TEXTS_DIR / f"{book_id}.txt").write_text(full_text, encoding="utf-8")

        # Also save per-chapter files
        book_dir = TEXTS_DIR / book_id
        book_dir.mkdir(exist_ok=True)
        for j, ch in enumerate(chapters):
            safe_name = re.sub(r'[^\w\-]', '_', ch["chapter"])[:60]
            (book_dir / f"{j:02d}_{safe_name}.txt").write_text(
                ch["text"], encoding="utf-8"
            )

    save_json(corpus, OUTPUT_DIR / "stage1_corpus.json")

    print("\nExtraction summary:")
    for book_id, data in corpus.items():
        s = data["stats"]
        print(f"  {book_id}: {s['num_chapters']} chapters, "
              f"{s['total_words']:,} words, {s['total_chars']:,} chars")

    print("\nStage 1 complete.")


if __name__ == "__main__":
    main()
