from pathlib import Path
from docling.document_converter import DocumentConverter
import time
import json


# ==========================================
# Project Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FOLDER = PROJECT_ROOT / "local" / "raw"
PARSED_FOLDER = PROJECT_ROOT / "local" / "parsed"

PARSED_FOLDER.mkdir(parents=True, exist_ok=True)


# ==========================================
# Check Input Folder
# ==========================================

print("=" * 70)
print(" PARSING PIPELINE")
print("=" * 70)

print(f"Raw folder    : {RAW_FOLDER}")
print(f"Parsed folder : {PARSED_FOLDER}")
print(f"Raw exists    : {RAW_FOLDER.exists()}")

print("=" * 70)
print()


# ==========================================
# Create Docling Converter
# ==========================================

print("Initializing Docling...")

converter = DocumentConverter()

print("Docling is ready!\n")


# ==========================================
# Get PDF Files
# ==========================================

pdf_files = sorted([f for f in RAW_FOLDER.glob("*") if f.is_file()])

total_files = len(pdf_files)

print("=" * 70)
print(f"TOTAL PDF FILES FOUND: {total_files}")
print("=" * 70)

if total_files == 0:

    print("No PDF files found!")
    print(f"Check this folder:")
    print(RAW_FOLDER)

else:

    print("Files:")

    for i, pdf in enumerate(pdf_files, start=1):
        print(f"  {i}. {pdf.name}")

print("=" * 70)
print()


# ==========================================
# Parse Every PDF
# ==========================================

successful = 0
failed = 0


for i, pdf_path in enumerate(pdf_files, start=1):

    start_time = time.time()

    print("\n" + "=" * 70)
    print(f" FILE {i}/{total_files}")
    print(f" Name: {pdf_path.name}")
    print("=" * 70)


    # ======================================
    # Output Paths
    # ======================================

    markdown_path = (
        PARSED_FOLDER /
        f"{pdf_path.stem}.md"
    )

    json_path = (
        PARSED_FOLDER /
        f"{pdf_path.stem}.json"
    )


    # ======================================
    # Skip Already Parsed Files
    # ======================================

    if markdown_path.exists() and json_path.exists():

        print("Already parsed → Skipping...")

        successful += 1

        continue


    try:

        print("Status: Parsing...")
        print("Please wait...")


        # ==================================
        # Parse PDF with Docling
        # ==================================

        result = converter.convert(
            str(pdf_path)
        )

        document = result.document

        print("Parsing finished.")


        # ==================================
        # Export Markdown
        # ==================================

        print("Exporting Markdown...")

        markdown = (
            document.export_to_markdown()
        )

        markdown_path.write_text(
            markdown,
            encoding="utf-8"
        )

        print(
            f"Markdown saved: "
            f"{markdown_path.name}"
        )


        # ==================================
        # Export Structured JSON
        # ==================================

        print("Exporting structured JSON...")

        document_json = (
            document.export_to_dict()
        )

        json_path.write_text(
            json.dumps(
                document_json,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            f"JSON saved: "
            f"{json_path.name}"
        )


        # ==================================
        # Timing & Progress
        # ==================================

        elapsed = time.time() - start_time

        successful += 1

        print(
            f"Time     : "
            f"{elapsed / 60:.2f} minutes"
        )

        print(
            f"Progress : "
            f"{i}/{total_files}"
        )


    except Exception as e:

        failed += 1
        elapsed = time.time() - start_time

        print("ERROR!")

        print(
            f"File: {pdf_path.name}"
        )

        print(
            f"Error: {e}"
        )

        print(
            f"Time before error: "
            f"{elapsed / 60:.2f} minutes"
        )


# ==========================================
# Final Summary
# ==========================================

print("\n")

print("=" * 70)
print(" PARSING FINISHED")
print("=" * 70)

print(f"Total files : {total_files}")
print(f"Successful  : {successful}")
print(f"Failed      : {failed}")

print("=" * 70)