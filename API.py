from fastapi import FastAPI
import requests
from PIL import Image
import torch 
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List
import torch.nn as nn
from pydantic import BaseModel

app = FastAPI()

class UrlsInput(BaseModel):
    urls: List[str]

def load_model(model, path):
    return model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

class DenseLayers(nn.Module):
  def __init__(self, feature_dim=25088):
        super().__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(in_features=self.feature_dim, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        self.last_layer = nn.Linear(4, 1)

  def forward(self, x):
      x = nn.functional.relu(self.fc1(x))
      x = nn.functional.relu(self.fc2(x))
      x = nn.functional.relu(self.fc3(x))
      x = nn.Sigmoid()(self.last_layer(x))
      return x

class NN(nn.Module):
  def __init__(self, in_channels, model_name):
    super().__init__()
    self.model_name = model_name
    self.model, self.out_dim_vgg16 = self.feature_extraction_vgg16(model_name=self.model_name)

  def feature_extraction_vgg16(self, model_name):

    # ---- Feature Extraction using vgg16 ----

    self.model = models.vgg16(pretrained=True)

    self.model = nn.Sequential(*list(self.model.features.children()))

    for params in self.model.parameters():
      params.requires_grad = False

    return self.model, 25088 # Output dimensions of vgg16.


  def forward(self, x):
    x = self.model(x)
    # output = self.my_NN_Denselayers(x, self.out_dim_vgg16)
    return x

path_for_feature_extration_model = r'E:\FYP\Email Brand Impersonation\Visual Detection-20240429T182630Z-001\Visual Detection\Siamese_Model\feature_Extraction_model.pth'
path_for_dense_model = r'E:\FYP\Email Brand Impersonation\Visual Detection-20240429T182630Z-001\Visual Detection\Siamese_Model\MY_Dense_model.pth'

feature_extraction_model = NN(3, 'vgg16')
feature_extraction_model.load_state_dict(torch.load(path_for_feature_extration_model, map_location=torch.device('cpu')))
Dense_model = DenseLayers(25088)
Dense_model.load_state_dict(torch.load(path_for_dense_model, map_location=torch.device('cpu')))
feature_extraction_model.eval()
Dense_model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def download_image(urls):
    images = ()
    
    response = requests.get(urls[0])
    first_image = Image.open(BytesIO(response.content))

    response = requests.get(urls[1])
    second_image = Image.open(BytesIO(response.content))

    return (first_image, second_image)

def preprocess_image(images):

    first_image = images[0]
    second_image = images[1]

    first_image.convert('RGB')
    first_image = transform(first_image)

    second_image.convert('RGB')
    second_image = transform(second_image)
    return (first_image, second_image)


@app.post('/visual-predict')
async def predict(input_data:UrlsInput):
    urls = input_data.urls
    list_of_urls = []
    output_lists = []
    output_dict = {}
    for i in range(1, len(urls)):
        list_of_urls.append((urls[0], urls[i]))
    
    for url_set in list_of_urls:
        images = download_image(url_set)
        tensors = preprocess_image(images)

        output1 = feature_extraction_model(tensors[0])
        output2 = feature_extraction_model(tensors[1])
        output1 = torch.flatten(output1, start_dim=0)
        output2 = torch.flatten(output2, start_dim=0)

        result = output1 - output2

        output = Dense_model(result)

        output = output.cpu().item()

        output_lists.append((url_set[0], url_set[1], output))

    for _, i in enumerate(output_lists):
       output_dict[_] = {'first_url' : i[0], 'second_url' : i[1], 'output' : i[2]}

    return output_dict
