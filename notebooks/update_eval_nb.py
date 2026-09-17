import json

file_path = '08_model_evaluation.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))
        if 'np.load(\'../results/preprocessed_data/X_test.npy\')' in src:
            new_src = '''from tensorflow.keras.utils import to_categorical
from npz_consolidator import load_consolidated_arrays

print("Loading test data...\\n")

# Load label encoding and metadata
with open('../results/preprocessed_data_efficient/label_encoding.json', 'r') as f:
    label_encoding = json.load(f)

with open('../results/preprocessed_data_efficient/image_metadata.json', 'r') as f:
    metadata = json.load(f)
    
with open('../results/preprocessed_data_efficient/split_indices.json', 'r') as f:
    split_info = json.load(f)

num_classes = len(label_encoding)
labels_all = np.array(metadata['labels'])
test_indices = split_info['test_indices']
y_test = to_categorical(labels_all[test_indices], num_classes=num_classes)

# Load test data
X_test = load_consolidated_arrays('../results/preprocessed_data_efficient', 'test')

reverse_encoding = {v: k for k, v in label_encoding.items()}
class_names = [reverse_encoding[i] for i in range(num_classes)]

print(f"✓ Test data loaded: {X_test.shape}")
print(f"✓ Number of classes: {num_classes}")

# Load models
print("\\nLoading trained models...\\n")

try:
    cnn_model = load_model('../models/cnn/cnn_baseline.h5')
    print("✓ CNN baseline model loaded")
except:
    cnn_model = None
    print("⚠️  CNN model not found")

try:
    mobilenet_model = load_model('../models/mobilenet/mobilenet_v2_final.h5')
    print("✓ MobileNet V2 model loaded")
except:
    mobilenet_model = None
    print("⚠️  MobileNet model not found")'''
            cell['source'] = [line + '\n' for line in new_src.split('\n')]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
