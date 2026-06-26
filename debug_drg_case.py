from pathlib import Path
import sys

from case_loader import load_case_file
from denial_extractor import normalize_ocr_text, extract_drg_before_after_rule_based

if len(sys.argv) < 2:
    print("Usage: python debug_drg_case.py <path_to_denial_pdf>")
    raise SystemExit(1)

file_path = Path(sys.argv[1])
text = normalize_ocr_text(load_case_file(str(file_path)))
before, after = extract_drg_before_after_rule_based(text)

print("DRG before:", before)
print("DRG after:", after)

lower = text.lower()
idx = lower.find("drg table")
if idx == -1:
    idx = lower.find("original codes billed")
if idx == -1:
    idx = lower.find("new coding assignment")

if idx != -1:
    print("
--- OCR text around DRG table ---")
    print(text[max(0, idx - 500): idx + 2500])
else:
    print("
No DRG Table/original codes/new coding assignment marker found in extracted text.")
