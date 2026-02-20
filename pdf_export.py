"""
pdf_export.py — Convert Markdown summary to styled PDF.

Uses pandoc (Markdown → HTML) + weasyprint (HTML → PDF) for reliable
Chinese text rendering with PingFang SC font on macOS.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


# CSS for PDF rendering
PDF_CSS = """
@page { size: A4; margin: 2cm; }
body {
    font-family: 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
    font-size: 12px; line-height: 1.8; color: #333;
}
h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 17px; color: #1a5276; margin-top: 20px; }
h3 { font-size: 14px; color: #2c3e50; margin-top: 14px; }
ul { padding-left: 18px; }
li { margin-bottom: 6px; }
a { color: #2980b9; text-decoration: none; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 11px; }
hr { border: none; border-top: 1px solid #ddd; margin: 16px 0; }
"""


def export_pdf(md_path: str) -> str | None:
    """
    Convert a Markdown file to PDF.

    Args:
        md_path: Path to the .md file

    Returns:
        Path to the generated .pdf file, or None if conversion failed.
    """
    md_path = str(md_path)
    pdf_path = md_path.rsplit(".", 1)[0] + ".pdf"

    # Check dependencies
    if not shutil.which("pandoc"):
        print("  ⚠️  pandoc not found. Install with: brew install pandoc")
        return None

    try:
        from weasyprint import HTML
    except ImportError:
        print("  ⚠️  weasyprint not installed. Install with: pip install weasyprint")
        return None

    try:
        # Step 1: Markdown → HTML via pandoc
        with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as tmp:
            tmp_html = tmp.name

        subprocess.run(
            [
                "pandoc", md_path,
                "-f", "markdown",
                "-t", "html5",
                "--standalone",
                "--embed-resources",
                "-o", tmp_html,
            ],
            check=True,
            capture_output=True,
        )

        # Step 2: Inject CSS
        html = Path(tmp_html).read_text(encoding="utf-8")
        html = html.replace("</head>", f"<style>{PDF_CSS}</style></head>")

        # Step 3: HTML → PDF via weasyprint
        HTML(string=html).write_pdf(pdf_path)

        return pdf_path

    except Exception as e:
        print(f"  ⚠️  PDF export error: {e}")
        return None

    finally:
        # Cleanup temp file
        try:
            os.unlink(tmp_html)
        except (OSError, UnboundLocalError):
            pass
