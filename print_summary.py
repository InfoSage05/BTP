import json

with open('pdf_summary_out.json', encoding='utf-8') as f:
    data = json.load(f)

with open('pdf_summaries_list.txt', 'w', encoding='utf-8') as out:
    for k, v in sorted(data.items()):
        out.write(f"==================================================\n")
        out.write(f"File: {k}\n")
        out.write(f"Pages: {v['page_count']} | Total Characters: {v['total_chars']}\n")
        out.write(f"First Page Excerpt:\n{v['first_page_head'][:1000].strip()}\n")
        out.write(f"Second Page Excerpt:\n{v['second_page_head'][:1000].strip()}\n\n")

print("Wrote pdf_summaries_list.txt")
