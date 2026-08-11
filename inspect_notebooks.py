import json

def inspect_notebook(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
        
    print(f"==================================================")
    print(f"NOTEBOOK: {nb_path}")
    print(f"Total cells: {len(nb['cells'])}")
    
    cell_summary = []
    for idx, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = "".join(cell.get('source', []))
        cell_summary.append({
            "idx": idx,
            "type": cell_type,
            "first_line": source.split('\n')[0] if source else "",
            "length": len(source)
        })
        if cell_type == 'markdown' and source.startswith('#'):
            print(f"  Cell {idx} [MD Header]: {source.split('\n')[0]}")
            
    return nb

nb1 = inspect_notebook('CHF_ML_Modeling.ipynb')
nb2 = inspect_notebook('CHF_Physics_Informed_Extensions.ipynb')
