from pathlib import Path
import re
import time


# ============================================================
# Folders
# ============================================================

PARSED_FOLDER = Path("local/parsed")
CLEANED_FOLDER = Path("local/cleaned")

CLEANED_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Headings that usually indicate the beginning of real paper content
CONTENT_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "aim",
    "aims",
    "objective",
    "objectives",
    "methods",
    "method",
    "materials and methods",
    "patients and methods",
    "patients",
    "study design",
    "case presentation",
    "case report",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "recommendations",
    "clinical presentation",
    "management",
    "treatment",
    "postoperative care",
    "follow-up",
}


# ============================================================
# Helper Functions
# ============================================================

def normalize_heading(text: str) -> str:
    """
    Normalize a Markdown heading for comparison.
    """

    text = text.strip()

    # Remove Markdown # symbols
    text = re.sub(r"^#+\s*", "", text)

    # Remove trailing #
    text = re.sub(r"\s+#*$", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


def is_markdown_heading(line: str) -> bool:
    """
    Check whether a line is a Markdown heading.
    """

    return bool(
        re.match(
            r"^\s{0,3}#{1,6}\s+\S+",
            line
        )
    )


def is_content_heading(line: str) -> bool:
    """
    Determine whether a Markdown heading represents
    meaningful paper content.
    """

    if not is_markdown_heading(line):
        return False

    heading = normalize_heading(line)

    # Exact match
    if heading in CONTENT_HEADINGS:
        return True

    # Handle headings such as:
    # "1. Introduction"
    # "2. Methods"
    # "3 Results"
    cleaned = re.sub(
        r"^\d+(\.\d+)*[\.\-\:\)]?\s*",
        "",
        heading
    )

    if cleaned in CONTENT_HEADINGS:
        return True

    # Partial matching for common section names
    for keyword in CONTENT_HEADINGS:

        if (
            cleaned.startswith(keyword + " ")
            or cleaned.endswith(" " + keyword)
        ):
            return True

    return False


def looks_like_author_line(line: str) -> bool:
    """
    Detect common author / affiliation / correspondence lines.
    This is intentionally conservative.
    """

    text = line.strip()

    if not text:
        return False

    lower = text.lower()

    # Common affiliation / author indicators
    keywords = [
        "department of",
        "faculty of",
        "university",
        "medical school",
        "hospital",
        "corresponding author",
        "email:",
        "e-mail:",
        "address:",
        "affiliation",
        "md",
        "phd",
        "frcs",
        "mbbs",
        "msc",
        "prof.",
        "professor",
    ]

    # Email
    if re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        text,
        re.IGNORECASE
    ):
        return True

    # Strong affiliation indicators
    if any(keyword in lower for keyword in keywords):
        return True

    return False


def looks_like_title(line: str) -> bool:
    """
    Detect a likely paper title.

    We only use this before the first meaningful section.
    """

    text = line.strip()

    if not text:
        return False

    # A Markdown heading is very likely to be the paper title
    if is_markdown_heading(text):
        return True

    return False


def remove_yaml_frontmatter(lines: list[str]) -> list[str]:
    """
    Remove YAML front matter if present.
    """

    if not lines:
        return lines

    if lines[0].strip() != "---":
        return lines
    for i in range(1, len(lines)):

        if lines[i].strip() == "---":
            return lines[i + 1:]

    return lines


# ============================================================
# Main Cleaning Logic
# ============================================================

def clean_markdown(markdown: str) -> str:
    """
    Clean parsed Markdown before chunking.

    Main goals:
        1. Remove front matter.
        2. Remove author / affiliation information.
        3. Remove standalone paper title.
        4. Preserve meaningful Markdown headings.
        5. Preserve all medical content.
        6. Keep paragraph boundaries.
    """

    # Normalize line endings
    markdown = markdown.replace("\r\n", "\n")
    markdown = markdown.replace("\r", "\n")

    lines = markdown.split("\n")

    # --------------------------------------------------------
    # Remove YAML front matter
    # --------------------------------------------------------

    lines = remove_yaml_frontmatter(lines)

    # --------------------------------------------------------
    # Remove excessive blank lines
    # --------------------------------------------------------

    cleaned_lines = []

    previous_blank = False

    for line in lines:

        if not line.strip():

            if not previous_blank:
                cleaned_lines.append("")

            previous_blank = True

        else:

            cleaned_lines.append(line.rstrip())

            previous_blank = False

    lines = cleaned_lines

    # --------------------------------------------------------
    # Find first meaningful content heading
    # --------------------------------------------------------

    first_content_heading_index = None

    for i, line in enumerate(lines):

        if is_content_heading(line):

            first_content_heading_index = i
            break

    # --------------------------------------------------------
    # If we found a real section:
    #
    # Everything before it is considered front matter.
    # --------------------------------------------------------

    if first_content_heading_index is not None:

        content_lines = lines[
            first_content_heading_index:
        ]

    else:

        # ----------------------------------------------------
        # Fallback:
        # If no known heading exists, remove obvious
        # author / affiliation/title lines conservatively.
        # ----------------------------------------------------

        content_lines = []

        front_matter = True

        for line in lines:

            stripped = line.strip()

            if not stripped:

                # Preserve blank lines after content begins
                if not front_matter:
                    content_lines.append("")

                continue

            # If a meaningful heading appears
            if is_content_heading(line):

                front_matter = False
                content_lines.append(line)
                continue

            if front_matter:

                if looks_like_author_line(line):
                    continue

                # Skip likely title before content
                if looks_like_title(line):
                    continue

                # Very short front-matter lines are ignored
                # only before actual content begins.
                if len(stripped) < 120:
                    continue

                # Otherwise assume content has started
                front_matter = False
                content_lines.append(line)

            else:

                content_lines.append(line)

    # --------------------------------------------------------
    # Remove empty heading sections
    # --------------------------------------------------------

    final_lines = []

    for i, line in enumerate(content_lines):

        if not line.strip():

            # Avoid excessive blank lines
            if final_lines and final_lines[-1] != "":
                final_lines.append("")

        else:

            final_lines.append(line.rstrip())
            # --------------------------------------------------------
    # Trim document
    # --------------------------------------------------------

    while final_lines and not final_lines[0].strip():
        final_lines.pop(0)

    while final_lines and not final_lines[-1].strip():
        final_lines.pop()

    return "\n".join(final_lines)


# ============================================================
# Process One File
# ============================================================

def process_file(input_path: Path) -> bool:

    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"Processing: {input_path.name}")
    print("=" * 70)

    try:

        # ----------------------------------------------------
        # Read parsed Markdown
        # ----------------------------------------------------

        markdown = input_path.read_text(
            encoding="utf-8"
        )

        original_chars = len(markdown)

        # ----------------------------------------------------
        # Clean
        # ----------------------------------------------------

        cleaned = clean_markdown(markdown)

        cleaned_chars = len(cleaned)

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_path = (
            CLEANED_FOLDER
            / input_path.name
        )

        output_path.write_text(
            cleaned,
            encoding="utf-8"
        )

        elapsed = time.time() - start_time

        print(
            f"Original characters : {original_chars:,}"
        )

        print(
            f"Cleaned characters  : {cleaned_chars:,}"
        )

        print(
            f"Removed characters  : "
            f"{original_chars - cleaned_chars:,}"
        )

        print(
            f"Saved: {output_path}"
        )

        print(
            f"Time: {elapsed:.2f} seconds"
        )

        return True

    except Exception as e:

        print("ERROR!")
        print(f"File: {input_path.name}")
        print(f"Error: {e}")

        return False


# ============================================================
# Main
# ============================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 70)
    print(" MEDICAL PAPER CLEANING")
    print("=" * 70)

    # --------------------------------------------------------
    # Find Markdown files
    # --------------------------------------------------------

    markdown_files = sorted(
        PARSED_FOLDER.glob("*.md")
    )

    total_files = len(markdown_files)

    print(
        f"TOTAL MARKDOWN FILES FOUND: "
        f"{total_files}"
    )

    if total_files == 0:

        print(
            "No Markdown files found!"
        )

        return

    print("\nFiles:")

    for i, file in enumerate(
        markdown_files,
        start=1
    ):

        print(
            f"  {i}. {file.name}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    successful = 0
    failed = 0

    for i, markdown_path in enumerate(
        markdown_files,
        start=1
    ):

        print(
            f"\nFILE {i}/{total_files}"
        )

        output_path = (
            CLEANED_FOLDER
            / markdown_path.name
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # We intentionally overwrite the cleaned file.
        # This allows us to rerun cleaning after modifying
        # the rules.
        # ----------------------------------------------------

        if process_file(markdown_path):
            successful += 1
        else:
            failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    elapsed = time.time() - start_time
    print("\n")
    print("=" * 70)
    print(" CLEANING FINISHED")
    print("=" * 70)

    print(
        f"Total files : {total_files}"
    )

    print(
        f"Successful  : {successful}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        f"Total time  : {elapsed / 60:.2f} minutes"
    )

    print("=" * 70)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()