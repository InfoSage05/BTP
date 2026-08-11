import os
import pypdf

pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]

with open("pdf_analysis.txt", "w", encoding="utf-8") as out:
    for f in sorted(pdf_files):
        try:
            reader = pypdf.PdfReader(f)
            out.write(f"=========================================\n")
            out.write(f"FILE: {f}\n")
            out.write(f"PAGE COUNT: {len(reader.pages)}\n")
            out.write(f"=========================================\n")
            
            # Print page 1 & 2 content excerpt
            for p_num in range(min(5, len(reader.pages))):
                txt = reader.pages[p_num].extract_text() or ""
                out.write(f"--- PAGE {p_num+1} ---\n")
                out.write(txt[:1000] + "\n\n")
        except Exception as e:
            out.write(f"ERROR reading {f}: {e}\n")

print("Done writing pdf_analysis.txt")
