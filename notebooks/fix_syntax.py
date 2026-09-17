import json
import re

notebook_path = 'd:/cnn_model/notebooks/11_prediction_testing.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell['source'])
        
        # Replace physical newlines inside strings that were accidentally introduced.
        # We can look for patterns like print("...\n\n")
        # In a notebook, an unescaped physical newline inside a string looks like:
        # "print(\"Loading test data from the current preprocessed split folders...\\n\n\",\n"
        # We want to replace it with:
        # "print(\"Loading test data from the current preprocessed split folders...\\n\")\n"
        
        source_str = source_str.replace('print("Loading test data from the current preprocessed split folders...\\n\n")', 'print("Loading test data from the current preprocessed split folders...\\n")')
        source_str = source_str.replace('print("\\n\nChecking availability of the three model files before evaluation...\\n\n")', 'print("\\nChecking availability of the three model files before evaluation...\\n")')
        source_str = source_str.replace('print("\\n\nMissing required model files:")', 'print("\\nMissing required model files:")')
        source_str = source_str.replace('print("\\n\nAll required models were found. Starting evaluation...\\n\n")', 'print("\\nAll required models were found. Starting evaluation...\\n")')
        source_str = source_str.replace('print("\\n\n✓ Saved: model_comparison.csv")', 'print("\\n✓ Saved: model_comparison.csv")')
        source_str = source_str.replace('print(f"\\n\n🏆 Best model selected: {best_model_name} ({accuracy:.4f})")', 'print(f"\\n🏆 Best model selected: {best_model_name} ({accuracy:.4f})")')
        source_str = source_str.replace('print("\\n\nModel comparison:")', 'print("\\nModel comparison:")')
        
        # Let's also do a general regex for any `\n` followed immediately by a newline inside `print("...\n\n")`
        source_str = re.sub(r'print\(\"([^\"]*?)\\n\n\"\)', r'print("\1\\n")', source_str)
        source_str = re.sub(r'print\(\"\\n\n([^\"]*?)\"\)', r'print("\\n\1")', source_str)
        source_str = re.sub(r'print\(\"\\n\n([^\"]*?)\\n\n\"\)', r'print("\\n\1\\n")', source_str)
        
        # Format strings:
        source_str = re.sub(r'print\(f\"([^\"]*?)\\n\n\"\)', r'print(f"\1\\n")', source_str)
        source_str = re.sub(r'print\(f\"\\n\n([^\"]*?)\"\)', r'print(f"\\n\1")', source_str)

        lines = source_str.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Syntax errors fixed.')
