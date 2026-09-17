import json

filepath = r"d:\cnn_model\notebooks\05_cnn_baseline.ipynb"
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

for idx, cell in enumerate(data.get("cells", [])):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "Loading preprocessed data using memory-efficient approach" in src:
            with open(r"d:\cnn_model\cell_src.txt", "w", encoding="utf-8") as f_out:
                f_out.write(src)
            print(f"Written cell source to cell_src.txt at index {idx}")
            break
