import os
import pypdf
import json

pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
pdf_summary = {}

for f in pdf_files:
    reader = pypdf.PdfReader(f)
    n_pages = len(reader.pages)
    
    pages_text = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ''
        pages_text.append(txt)
        
    full_text = "\n".join(pages_text)
    
    # Extract first page text (title, abstract, authors)
    first_page = pages_text[0] if len(pages_text) > 0 else ""
    second_page = pages_text[1] if len(pages_text) > 1 else ""
    
    pdf_summary[f] = {
        "filename": f,
        "page_count": n_pages,
        "first_page_head": first_page[:1500],
        "second_page_head": second_page[:1500],
        "total_chars": len(full_text)
    }

with open("pdf_summary_out.json", "w", encoding="utf-8") as out:
    json.dump(pdf_summary, out, indent=2, ensure_ascii=False)

print("Extracted PDF summaries for:", list(pdf_summary.keys()))
