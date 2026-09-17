import json
import re

notebook_path = 'd:/cnn_model/notebooks/11_prediction_testing.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Code for the get_predictions_in_batches function
batch_func_code = """
def get_predictions_in_batches(X, model, model_path, batch_size=32):
    predictions = []
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size].copy()
        batch = prepare_inputs_for_model(batch, model, model_path)
        preds = model.predict(batch, verbose=0)
        predictions.append(preds)
    return np.vstack(predictions)
"""

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = ''.join(cell['source'])
        
        # 1. Inject the get_predictions_in_batches function right after prepare_inputs_for_model
        if 'def prepare_inputs_for_model' in source_str:
            if 'def get_predictions_in_batches' not in source_str:
                new_source_str = source_str + "\n" + batch_func_code
                lines = new_source_str.split('\n')
                cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
        
        # 2. Update the evaluation loop
        elif 'for spec in resolved_model_specs:' in source_str:
            # Replace:
            # eval_images = prepare_inputs_for_model(X_test, model, str(model_path))
            # predictions = model.predict(eval_images, verbose=0)
            # With:
            # predictions = get_predictions_in_batches(X_test, model, str(model_path))
            
            source_str = re.sub(
                r'eval_images\s*=\s*prepare_inputs_for_model\(X_test,\s*model,\s*str\(model_path\)\)\s*\n\s*predictions\s*=\s*model\.predict\(eval_images,\s*verbose=0\)',
                r'predictions = get_predictions_in_batches(X_test, model, str(model_path))',
                source_str
            )
            lines = source_str.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]
            
        # 3. Update the best model prediction
        elif 'prepared_images = prepare_inputs_for_model' in source_str:
            # Replace:
            # prepared_images = prepare_inputs_for_model(X_test, model, str(best_model_path))
            # 
            # # Predict
            # predictions = model.predict(prepared_images, verbose=0)
            # With:
            # predictions = get_predictions_in_batches(X_test, model, str(best_model_path))
            
            source_str = re.sub(
                r'prepared_images\s*=\s*prepare_inputs_for_model\(X_test,\s*model,\s*str\(best_model_path\)\)\s*\n\s*(?:#.*?\n\s*)?predictions\s*=\s*model\.predict\(prepared_images,\s*verbose=0\)',
                r'predictions = get_predictions_in_batches(X_test, model, str(best_model_path))',
                source_str
            )
            lines = source_str.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines[-1] else [line + '\n' for line in lines[:-1]]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated 11_prediction_testing.ipynb with batch prediction successfully!')
