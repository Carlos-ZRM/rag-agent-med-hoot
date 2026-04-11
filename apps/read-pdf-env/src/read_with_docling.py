"""
read_with_docling.py
--------------------
Read PDF files and extract plain text using Docling.

Usage:
    python read_with_docling.py <file.pdf>
    python read_with_docling.py <file.pdf> --output result.txt

Examples:
    python read_with_docling.py report.pdf
    python read_with_docling.py report.pdf --output extracted.txt
"""

import argparse
import sys
from pathlib import Path


def load_docling():
    """Import Docling with a helpful error if not installed."""
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        return DocumentConverter, PdfPipelineOptions
    except ImportError:
        print("Error: Docling is not installed.")
        print("Install it with:  poetry add docling  or  pip install docling")
        sys.exit(1)


def read_pdf_as_text(pdf_path: str) -> str:
    """Convert a PDF file to plain text using Docling."""
    DocumentConverter, PdfPipelineOptions = load_docling()

    # Configure pipeline: enable OCR for scanned PDFs
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = True
    pdf_opts.do_table_structure = True

    converter = DocumentConverter()

    print(f"Reading: {pdf_path}", file=sys.stderr)
    result = converter.convert(pdf_path)

    return result.document.export_to_text()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract plain text from a PDF using Docling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "pdf",
        help="Path to the PDF file.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Optional output .txt file. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Warning: File does not have a .pdf extension: {pdf_path}", file=sys.stderr)

    text = read_pdf_as_text(str(pdf_path))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved to: {out_path}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()