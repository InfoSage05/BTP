import json

def dump_notebook_details(nb_path, out_file):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
        
    out_file.write(f"\n==================================================\n")
    out_file.write(f"NOTEBOOK: {nb_path}\n")
    out_file.write(f"Total cells: {len(nb['cells'])}\n")
    out_file.write(f"==================================================\n\n")
    
    for idx, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = "".join(cell.get('source', []))
        outputs = cell.get('outputs', [])
        
        out_file.write(f"--- CELL {idx} ({cell_type}) ---\n")
        out_file.write(source + "\n")
        
        if outputs:
            out_file.write(f"=== OUTPUTS ({len(outputs)}) ===\n")
            for out_idx, out in enumerate(outputs):
                output_type = out.get('output_type', '')
                if output_type == 'stream':
                    text = "".join(out.get('text', []))
                    out_file.write(f"[Stream text]:\n{text[:1500]}\n")
                elif output_type == 'execute_result' or output_type == 'display_data':
                    data = out.get('data', {})
                    if 'text/plain' in data:
                        text = "".join(data['text/plain'])
                        out_file.write(f"[Execute result text/plain]:\n{text[:1500]}\n")
                    if 'text/html' in data:
                        out_file.write(f"[Execute result text/html present]\n")
                elif output_type == 'error':
                    ename = out.get('ename', '')
                    evalue = out.get('evalue', '')
                    out_file.write(f"[Error]: {ename}: {evalue}\n")
        out_file.write("\n" + "-"*40 + "\n\n")

with open("notebooks_dump.txt", "w", encoding="utf-8") as out:
    dump_notebook_details('CHF_ML_Modeling.ipynb', out)
    dump_notebook_details('CHF_Physics_Informed_Extensions.ipynb', out)

print("Dumped notebooks_dump.txt successfully")
