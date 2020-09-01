import io
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image
from torch import nn, optim
import base64

app = Flask(__name__)

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((28, 28)),
                                        transforms.CenterCrop((28, 28)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.5),
                                            (0.5))])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction_MNIST(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    tensor = tensor.view(1, 784)
    model = torch.load('/mnt/i/ML/ML-2/server/MNIST.pth')
    with torch.no_grad():
        logits = model.forward(tensor)
    prob = F.softmax(logits, dim=1)
    y = prob.detach()[0].numpy()
    x = [x for x in range(10)]
    dict = {}
    for a, b in zip(x, y):
        dict[a] = b
    max_key = max(dict, key=lambda k: dict[k])
    return max_key, dict[max_key]

@app.route('/api/MNIST/FFN', methods=['POST'])
def MNIST():
    if request.method == 'POST':
        if request.files['file'].filename == '':
            return jsonify({'error': 'No Image file found!'})
        file = request.files['file']
        img_bytes = file.read()
        class_id, probability = get_prediction_MNIST(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'probability': str(probability)})


def transform_draw_image(image):
    my_transforms = transforms.Compose([transforms.Resize((28, 28), interpolation=Image.NEAREST),
                                        transforms.CenterCrop((28, 28)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            (0.5),
                                            (0.5))])
    return my_transforms(image).unsqueeze(0)

def get_prediction_MNIST_Draw(image):
    tensor = transform_draw_image(image=image)
    tensor = tensor.view(1, 784)
    model = torch.load('/mnt/i/ML/ML-2/server/MNIST.pth')
    with torch.no_grad():
        logits = model.forward(tensor)
    prob = F.softmax(logits, dim=1)
    y = prob.detach()[0].numpy()
    x = [x for x in range(10)]
    dict = {}
    for a, b in zip(x, y):
        dict[a] = b
    max_key = max(dict, key=lambda k: dict[k])
    return max_key, dict[max_key]

@app.route('/api/MNIST/DRW', methods=['POST'])
def MNIST_DRAW():
    data = request.json
    im = Image.open(io.BytesIO(base64.b64decode(data['bs64'])))
    class_id, probability = get_prediction_MNIST_Draw(image=im)
    return jsonify({'class_id': class_id, 'probability': str(probability)})


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
        
def get_prediction_MNIST_(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    model = Net()
    model.load_state_dict(torch.load('/mnt/i/ML/ML-2/server/MNIST_state_dict.pth'))
    with torch.no_grad():
        logits = model.forward(tensor)
    # print(logits)
    prob = torch.exp(logits)
    y = prob.detach()[0].numpy()
    x = [x for x in range(10)]
    dict = {}
    for a, b in zip(x, y):
        dict[a] = b
    max_key = max(dict, key=lambda k: dict[k])
    return max_key, dict[max_key]


@app.route('/api/MNIST/CNN', methods=['POST'])
def MNIST_():
    if request.method == 'POST':
        if request.files['file'].filename == '':
            return jsonify({'error': 'No Image file found!'})
        file = request.files['file']
        img_bytes = file.read()
        class_id, probability = get_prediction_MNIST_(image_bytes=img_bytes) 
        return jsonify({'class_id': class_id, 'probability': str(probability)})

#Drw

def get_prediction_MNIST_Draw_CNN(image):
    tensor = transform_draw_image(image)
    model = Net()
    model.load_state_dict(torch.load('/mnt/i/ML/ML-2/server/MNIST_state_dict.pth'))
    with torch.no_grad():
        logits = model.forward(tensor)
    # print(logits)
    prob = torch.exp(logits)
    y = prob.detach()[0].numpy()
    x = [x for x in range(10)]
    dict = {}
    for a, b in zip(x, y):
        dict[a] = b
    max_key = max(dict, key=lambda k: dict[k])
    return max_key, dict[max_key]


@app.route('/api/MNIST/Draw/CNN', methods=['POST'])
def MNIST_CNN():
    data = request.json
    im = Image.open(io.BytesIO(base64.b64decode(data['bs64'])))
    class_id, probability = get_prediction_MNIST_Draw_CNN(image=im)
    return jsonify({'class_id': class_id, 'probability': str(probability)})