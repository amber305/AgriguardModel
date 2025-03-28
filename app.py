from flask import Flask, request, jsonify
import torch
import torchvision.transforms as T
from PIL import Image
import os
from flask_cors import CORS

# Load the model
MODEL_PATH = 'model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
import timm
model = timm.create_model("rexnet_150", pretrained=False, num_classes=17)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

model.to(device)
model.eval()

# Define transformations
mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
tfs = T.Compose([
    T.Resize((im_size, im_size)),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])

# Class names
classes = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight', 'Rice___Brown_Spot',
    'Rice___Healthy', 'Rice___Leaf_Blast', 'Rice___Neck_Blast', 'Sugarcane_Bacterial Blight',
    'Sugarcane_Healthy', 'Sugarcane_Red Rot', 'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
]

# Flask app
app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        img = Image.open(file).convert('RGB')
        img = tfs(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][pred].item()

        return jsonify({'class': classes[pred], 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Model is ready to predict'})

if __name__ == '__main__':
    po = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=po)
