import torch
from torchvision import models, datasets, transforms
from torchvision.io import read_image
import os
import sys

dir = sys.argv[1]
pre = sys.argv[2]

preffix = './resnet_predictions'
if not os.path.isdir(preffix):
        os.mkdir(preffix)
preffix += '/' + dir
if not os.path.isdir(preffix):
        os.mkdir(preffix)

# Initialize the model
weights = models.ResNet50_Weights.DEFAULT
resnet = models.resnet50(weights=weights)

# # Important: put network into evaluation mode
# # Some networks have layers that do not behave the same during train/eval
# # Forgetting this is a very common source of bugs
resnet.eval()

preprocess = weights.transforms()

subdir = os.listdir('./data/' + dir + '-opencv')
tensors = {} 
for img_name in subdir:
    if img_name.startswith('frame' + pre):
        img = read_image('./data/' + dir + '-opencv/' + img_name)
        print(img_name)
        batch = preprocess(img).unsqueeze(0)
        prediction = resnet(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = weights.meta["categories"][class_id]
        tensors[img_name.split('.')[0]] = prediction

print(f'Number of files: {len(subdir)}')
print(f'Resnets: {len(tensors)}')

if len(tensors) > 0:
    torch.save(tensors, preffix + '/' + dir + pre + '.pth')