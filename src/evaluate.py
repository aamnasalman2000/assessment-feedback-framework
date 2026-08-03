from pathlib import Path
import fitz

pdf_path = fitz.open("data\\submissions\\75_Report.pdf")

doc = fitz.open(pdf_path)

text = ""
for page in doc:
    text += page.get_text()

output_path = Path("data/extracted")
output_path.mkdir(parents=True, exist_ok=True)

(output_path / "75_Report.txt").write_text(text, encoding="utf-8")

print("Extraction complete!")