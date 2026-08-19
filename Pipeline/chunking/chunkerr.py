from pathlib import Path
import json
import time
import re

from docling_core.types.doc import DoclingDocument
from docling_core.transforms.chunker import HybridChunker


# ============================================================
# Folders
# ============================================================

PARSED_FOLDER = Path("local/parsed")
CHUNKS_FOLDER = Path("local/chunks")

CHUNKS_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# Chunking Configuration
# ============================================================

# Larger chunks than the previous version.
MAX_TOKENS = 800

# Approximate contextual overlap.
# We use neighboring chunk text instead of a fake
# HybridChunker "overlap" parameter.
OVERLAP_RATIO = 0.25

# Ignore very small front-matter chunks.
MIN_CONTENT_WORDS = 25

# Only look for author/affiliation lines within the first N
# characters of a chunk (avoids false positives deep in the body).
FRONT_MATTER_SEARCH_WINDOW = 700


# ============================================================
# Initialize Docling HybridChunker
# ============================================================

print("Initializing Docling HybridChunker...")

chunker = HybridChunker(
    max_tokens=MAX_TOKENS,
    merge_peers=True,
)

print("Docling HybridChunker is ready!\n")


# ============================================================
# Utility Functions
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace without destroying document structure."""

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def word_count(text: str) -> int:
    """Return approximate number of words."""

    return len(text.split())


def get_overlap_text(
    text: str,
    ratio: float
) -> str:
    """
    Return the last part of a neighboring chunk.

    This gives the next chunk some contextual overlap.

    NOTE: we slice the ORIGINAL string by character position (found via
    \\S+ word boundaries) instead of `" ".join(text.split()[-n:])`.
    The old join-based approach silently flattened every newline inside
    the overlap into a single space, so the borrowed context read as one
    run-on line while the rest of the chunk kept its original paragraph
    breaks. Slicing preserves the source formatting.
    """

    matches = list(re.finditer(r"\S+", text))

    if not matches:
        return ""

    overlap_words = max(
        1,
        int(len(matches) * ratio)
    )

    start_char = matches[-overlap_words].start()

    return text[start_char:].strip()


def get_forward_overlap_text(
    text: str,
    ratio: float
) -> str:
    """
    Return the first part of a neighboring chunk.

    This gives the previous chunk some forward context.
    See note in get_overlap_text() about why we slice by character
    position instead of re-joining split words.
    """

    matches = list(re.finditer(r"\S+", text))

    if not matches:
        return ""

    overlap_words = max(
        1,
        int(len(matches) * ratio)
    )

    end_char = matches[overlap_words - 1].end()

    return text[:end_char].strip()


# ============================================================
# Shared Front-Matter Detection Helpers
# ============================================================

# Deliberately NOT using re.VERBOSE here: in VERBOSE mode Python strips
# unescaped whitespace out of the *entire* pattern, including inside
# plain literal text. That silently turned "Department of" into
# "Departmentof" in an earlier draft of this pattern, which meant some
# affiliation lines (anything relying only on the "Department of"
# alternative, with no "University"/"MD" elsewhere on the same line)
# were never stripped. Kept as a flat list of alternatives instead.
FRONT_MATTER_LINE = re.compile(
    "|".join([
        r".*\b(?:MD|MBBCh|FRCS|FACS|MRCS|PhD|Prof\.?)\b.*",
        r".*\bDepartment of\b.*",
        r".*\bFaculty of\b.*",
        r".*\bUniversity\b.*",
        r".*\bInstitute\b.*",
        r".*\bCorresponding author\b.*",
        r".*\be-?mail\s*:.*",
        r".*\bMobile\s*:?\s*\d{5,}.*",
        r"^[-*]?\s*\d*\s*(?:Department|Faculty|Division|Unit|Pediatric|Community)\b.*",
    ]),
    re.IGNORECASE,
)

# A line that mentions common abstract/body vocabulary is almost
# certainly a real sentence, not an author list — even if it happens
# to contain two capitalized words in a row (place names, eponyms...).
_ABSTRACT_STOPWORDS = re.compile(
    r"\b(the|is|was|were|this|these|study|which|that|patients|we|our)\b",
    re.IGNORECASE,
)

# Where the real paper content typically starts. Everything before the
# first match of this, at the top of the FIRST chunk of a document, is
# a candidate for stripping.
_ABSTRACT_START = re.compile(
    r"(Background|Abstract|Introduction(?:\s+and\s+objectives?)?"
    r"|Objective|Objectives|Aim|Purpose|Methods)\s*:",
    re.IGNORECASE,
)


def _looks_like_author_line(line: str) -> bool:
    """
    Fallback for author-name lines that don't contain any of the
    Department/University/MD keywords — e.g. a bare
    "Firstname Lastname 1, Firstname Lastname 2 and Firstname Lastname 3"
    or "* Ahmed Alaa Eldin Abd El Raouf Khorshid, ** Ismail Abdel Hakim
    Kotb ... and ** Kamal Mamdouh Kamal Elsaid.".
    """

    words = line.split()

    if not words or len(words) > 30:
        return False

    if _ABSTRACT_STOPWORDS.search(line):
        return False

    name_pairs = len(
        re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", line)
    )

    has_list_structure = ("," in line) or (" and " in line.lower())

    return name_pairs >= 2 and has_list_structure


def _looks_like_contact_line(line: str) -> bool:
    """
    Catches short "Name: Mobile 012345\\ne-mail: x@y.com" style contact
    lines that have neither an author-list structure nor the
    Department/University keywords FRONT_MATTER_LINE looks for.
    """

    if _ABSTRACT_STOPWORDS.search(line):
        return False

    has_contact_marker = bool(
        re.search(r"\bmobile\b|\be-?mail\s*:|\btel(?:ephone)?\b|@", line, re.IGNORECASE)
    )

    return has_contact_marker and len(line.split()) <= 12


def is_front_matter_line(line: str) -> bool:
    """Single source of truth used by both the whole-chunk and
    prefix-stripping checks below."""

    return (
        bool(FRONT_MATTER_LINE.match(line))
        or _looks_like_author_line(line)
        or _looks_like_contact_line(line)
    )


# ============================================================
# Detect Low-Value Front Matter
# ============================================================

def is_low_value_front_matter(
    content: str,
    headings: list[str]
) -> bool:
    """
    Detect chunks that are ENTIRELY paper metadata: authors,
    departments, affiliations, correspondence/contact details.

    Rewritten to classify LINE BY LINE instead of gating on total
    word count + a short keyword list. The old version had two real
    gaps that showed up once MAX_TOKENS was raised (HybridChunker
    started putting the author block in its own chunk, separate from
    the abstract, instead of merging them):
      1. `if word_count(content) >= MIN_CONTENT_WORDS: return False`
         meant a multi-line affiliation block (e.g. 3 institutions +
         a correspondence line, ~60 words) was skipped outright just
         for being "long enough", even though every line in it was
         metadata.
      2. Pure name lists with no "MD"/"Department"/"University" token
         (e.g. "* Ahmed Alaa Eldin Abd El Raouf Khorshid, ** Ismail
         Abdel Hakim Kotb ...") or contact lines using "e-mail:"/
         "Mobile" (not in the old keyword list, or blocked by a
         hyphen mismatch against "email:") scored 0 keyword hits and
         were never flagged.
    We DO NOT remove normal headings from useful sections.
    """

    text = content.strip()

    if not text:
        return True

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if lines and all(is_front_matter_line(line) for line in lines):
        return True

    # Very short title-only chunks (chunk text == its own heading).
    if headings:
        heading = headings[-1].strip().lower()

        if text.lower() == heading:
            return True

    return False


# ============================================================
# Strip Author / Affiliation Lines From a Chunk's Prefix
# ============================================================

def strip_leading_front_matter(content: str) -> tuple[str, str]:
    """
    Remove author/affiliation/correspondence lines that appear BEFORE
    the actual abstract text (Background:/Abstract:/Introduction:/...).

    This is the fix for the actual complaint: author names and
    "Department of ..., University, Egypt" lines were ending up glued
    to the start of the abstract chunk. is_low_value_front_matter()
    only removes a chunk that is metadata *in its entirety* — it can't
    trim a prefix off an otherwise-good chunk, which is what's needed
    when the author block and abstract share one chunk.

    Returns (cleaned_content, removed_text). removed_text is kept in
    metadata rather than thrown away, so nothing is silently lost.
    """

    match = _ABSTRACT_START.search(
        content[:FRONT_MATTER_SEARCH_WINDOW]
    )

    if not match or not content[:match.start()].strip():
        return content, ""

    head = content[:match.start()]
    body = content[match.start():]

    head_lines = [line for line in head.split("\n") if line.strip()]

    kept_lines = []
    removed_lines = []

    for line in head_lines:
        stripped = line.strip()

        if is_front_matter_line(stripped):
            removed_lines.append(line)
        else:
            kept_lines.append(line)

    cleaned = (
        ("\n".join(kept_lines) + "\n" if kept_lines else "")
        + body
    ).strip()

    removed = "\n".join(removed_lines)

    return cleaned, removed


def pick_topic(headings: list[str]) -> str:
    """
    Pick the most useful heading to use as `topic`.

    Docling sometimes mislabels the author/affiliation line as the
    document's only "heading", which meant `topic` ended up being
    something like "Tamer M Said Salama, MD; Mohamed Ibrahim Hassan, MD"
    instead of the paper's actual title or section name. We walk the
    headings from most to least specific and skip anything that looks
    like an author line or a bare "Correspondence:" label.
    """

    generic_labels = {"correspondence:", "correspondence"}

    for heading in reversed(headings):
        candidate = heading.strip()

        if not candidate:
            continue

        if candidate.lower() in generic_labels:
            continue

        if is_front_matter_line(candidate):
            continue

        return candidate

    # Nothing usable — most likely docling only ever saw the author
    # line as a heading for this document. Better to say so explicitly
    # than to silently return an author name as the "topic".
    return "Unknown"


# ============================================================
# Build Metadata
# ============================================================

def build_chunk_metadata(
    chunk,
    document_id: str,
    source_pdf: str,
    chunk_index: int,
    total_chunks: int,
):
    """
    Extract metadata and provenance directly from Docling.
    """

    doc_items = []
    page_numbers = set()
    for item in chunk.meta.doc_items:

        item_data = {
            "self_ref": item.self_ref,

            "label": (
                str(item.label)
                if item.label
                else None
            ),

            "content_layer": (
                str(item.content_layer)
                if item.content_layer
                else None
            ),

            "prov": [],
        }

        for prov in item.prov:

            page_no = getattr(
                prov,
                "page_no",
                None
            )

            if page_no is not None:
                page_numbers.add(page_no)

            bbox = getattr(
                prov,
                "bbox",
                None
            )

            bbox_data = None

            if bbox is not None:
                bbox_data = {
                    "l": bbox.l,
                    "t": bbox.t,
                    "r": bbox.r,
                    "b": bbox.b,
                    "coord_origin": str(
                        bbox.coord_origin
                    ),
                }

            charspan = getattr(
                prov,
                "charspan",
                None
            )

            prov_data = {
                "page_no": page_no,
                "bbox": bbox_data,
                "charspan": (
                    list(charspan)
                    if charspan
                    else None
                ),
            }

            item_data["prov"].append(
                prov_data
            )

        doc_items.append(item_data)

    # --------------------------------------------------------
    # Headings
    # --------------------------------------------------------

    headings = []

    if hasattr(chunk.meta, "headings"):
        headings = list(
            chunk.meta.headings or []
        )

    topic = pick_topic(headings)

    # --------------------------------------------------------
    # Final metadata
    # --------------------------------------------------------

    return {
        "document_id": document_id,

        # Actual source document
        "source": source_pdf,

        # Structure
        "topic": topic,
        "headings": headings,

        # Chunk information
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,

        # Provenance
        "pages": sorted(page_numbers),

        "docling": {
            "doc_items": doc_items,
        },
    }


# ============================================================
# Process One Document
# ============================================================

def process_document(json_path: Path):

    start_time = time.time()

    document_id = json_path.stem
    source_pdf = f"{document_id}.pdf"

    print("\n" + "=" * 70)
    print(f"Processing: {source_pdf}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Docling JSON
    # --------------------------------------------------------

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        document_data = json.load(f)

    # --------------------------------------------------------
    # Reconstruct DoclingDocument
    # --------------------------------------------------------

    document = DoclingDocument.model_validate(
        document_data
    )

    print("DoclingDocument loaded.")

    # --------------------------------------------------------
    # Hybrid Chunking
    # --------------------------------------------------------

    raw_chunks = list(
        chunker.chunk(
            dl_doc=document
        )
    )

    print(
        f"HybridChunker generated: "
        f"{len(raw_chunks)} chunks"
    )

    # --------------------------------------------------------
    # First pass:
    # normalize + remove useless front matter
    # --------------------------------------------------------

    filtered_chunks = []

    for position, chunk in enumerate(raw_chunks):
        content = normalize_text(
            chunk.text
        )

        if not content:
            continue

        headings = []

        if hasattr(chunk.meta, "headings"):
            headings = list(
                chunk.meta.headings or []
            )

        # Remove author/title/affiliation-only chunks
        # (whole chunk is metadata, nothing else).
        if is_low_value_front_matter(
            content,
            headings
        ):
            print(
                "Skipping low-value front matter:"
            )

            print(
                f"  {content[:120]}..."
            )

            continue

        # Strip author/affiliation lines glued to the front of the
        # FIRST real chunk (the far more common case: metadata sharing
        # a chunk with the abstract, not its own separate chunk).
        removed_front_matter = ""

        if position == 0:
            content, removed_front_matter = (
                strip_leading_front_matter(content)
            )

            if removed_front_matter:
                print(
                    "Stripped leading front matter:"
                )

                print(
                    f"  {removed_front_matter[:160]}"
                )

        filtered_chunks.append(
            {
                "chunk": chunk,
                "content": content,
                "headings": headings,
                "removed_front_matter": removed_front_matter,
            }
        )

    print(
        f"After front-matter filtering: "
        f"{len(filtered_chunks)} chunks"
    )

    # --------------------------------------------------------
    # Second pass:
    # Add contextual overlap
    # --------------------------------------------------------

    chunks = []

    for index, item in enumerate(
        filtered_chunks
    ):

        original_content = item["content"]

        previous_content = ""

        next_content = ""

        # Previous chunk overlap
        if index > 0:

            previous_content = (
                get_overlap_text(
                    filtered_chunks[
                        index - 1
                    ]["content"],
                    OVERLAP_RATIO,
                )
            )

        # Next chunk overlap
        if index < len(
            filtered_chunks
        ) - 1:

            next_content = (
                get_forward_overlap_text(
                    filtered_chunks[
                        index + 1
                    ]["content"],
                    OVERLAP_RATIO,
                )
            )

        # ----------------------------------------------------
        # Build final content
        # ----------------------------------------------------

        final_parts = []

        if previous_content:
            final_parts.append(
                "[Previous context]\n"
                + previous_content
            )

        final_parts.append(
            original_content
        )

        if next_content:
            final_parts.append(
                "[Next context]\n"
                + next_content
            )

        final_content = "\n\n".join(
            final_parts
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        metadata = build_chunk_metadata(
            chunk=item["chunk"],
            document_id=document_id,
            source_pdf=source_pdf,
            chunk_index=index + 1,
            total_chunks=len(
                filtered_chunks
            ),
        )

        metadata["overlap"] = {
            "enabled": True,
            "ratio": OVERLAP_RATIO,
            "previous_context": bool(
                previous_content
            ),
            "next_context": bool(
                next_content
            ),
        }

        # Kept for traceability instead of silently discarding it —
        # lets you audit what got stripped without re-running the parser.
        if item.get("removed_front_matter"):
            metadata["removed_front_matter"] = (
                item["removed_front_matter"]
            )

        chunks.append(
            {
                "content": final_content,
                "metadata": metadata,
            }
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        CHUNKS_FOLDER
        / f"{document_id}_chunks.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    elapsed = time.time() - start_time

    print(
        f"Saved: {output_path.name}"
    )

    print(
        f"Final chunks: {len(chunks)}"
    )

    print(
        f"Time: {elapsed / 60:.2f} minutes"
    )

    return len(chunks)


# ============================================================
# Main
# ============================================================

def main():
    start_time = time.time()

    print("\n")
    print("=" * 70)
    print(" MEDICAL RAG CHUNKING")
    print("=" * 70)

    json_files = sorted(
        PARSED_FOLDER.glob("*.json")
    )

    total_documents = len(json_files)

    print(
        f"TOTAL DOCUMENTS FOUND: "
        f"{total_documents}"
    )

    if total_documents == 0:

        print(
            "No parsed JSON files found!"
        )

        return

    print("\nFiles:")

    for i, file in enumerate(
        json_files,
        start=1
    ):

        print(
            f"  {i}. {file.name}"
        )

    print("=" * 70)

    successful = 0
    failed = 0
    total_chunks = 0

    for i, json_path in enumerate(
        json_files,
        start=1
    ):

        print(
            f"\nDOCUMENT {i}/"
            f"{total_documents}"
        )

        output_path = (
            CHUNKS_FOLDER
            / f"{json_path.stem}_chunks.json"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Delete old chunks before running this new version.
        # ----------------------------------------------------

        if output_path.exists():

            print(
                "Existing chunk file found."
            )

            print(
                "Re-generating with the new "
                "chunking strategy..."
            )

            output_path.unlink()

        try:

            count = process_document(
                json_path
            )

            total_chunks += count

            successful += 1

        except Exception as e:

            failed += 1

            print("ERROR!")

            print(
                f"File: {json_path.name}"
            )

            print(
                f"Error: {e}"
            )

    # ========================================================
    # Final Summary
    # ========================================================

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 70)
    print(" CHUNKING FINISHED")
    print("=" * 70)

    print(
        f"Total documents : "
        f"{total_documents}"
    )

    print(
        f"Successful      : "
        f"{successful}"
    )

    print(
        f"Failed          : "
        f"{failed}"
    )

    print(
        f"Total chunks    : "
        f"{total_chunks}"
    )

    print(
        f"Total time      : "
        f"{elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()