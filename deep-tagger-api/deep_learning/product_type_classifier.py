import torch
from torch import nn
from PIL import Image
from torchvision import transforms
import numpy as np

#Copy TinyVGG architecture from https://poloclub.github.io/cnn-explainer/
class FashionModelV2(nn.Module):
  def __init__(self,
               input_shape,
               hidden_units,
               output_shape):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)) #reduce 28x28 to 14x14
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)) # reduce 14x14 to 7x7
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=hidden_units*7*7, out_features=output_shape))

  def forward(self,x):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.classifier(x)
    return x

MODEL_PATH = 'deep_learning/torch_state/fashion_model_classifier_tiny_vgg_v2.pth'
CLASS_NAMES = ['T-shirt/top', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']
torch_classifier_model = FashionModelV2(input_shape=1, hidden_units=10, output_shape=10)
torch_classifier_model.load_state_dict(torch.load(f=MODEL_PATH, map_location=torch.device('cpu')))
torch_classifier_model.eval()


def _to_fashion_mnist(img_array: np.ndarray):
  """
  Transform an image array to Fashion MNIST format.

  Fashion MNIST specs:
  - 28x28 pixels
  - Grayscale (1 channel)
  - Normalized with mean=0.2860, std=0.3530 (standard Fashion MNIST stats)
  - Shape: [1, 1, 28, 28] (batch, channels, height, width)
  """
  image = Image.fromarray(img_array)

  # Fashion MNIST has white clothing on black background.
  # Most product photos are the opposite (dark item on white bg),
  # so we optionally invert. Toggle this based on your source images.
  invert = True

  # Build the transform pipeline
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # -> 1 channel
    transforms.Resize((28, 28)),  # -> 28x28
    transforms.RandomInvert(p=1.0 if invert else 0.0),  # invert if needed
    transforms.ToTensor(),  # -> [0,1], shape [1,28,28]
    transforms.Normalize((0.2860,), (0.3530,)),  # Fashion MNIST stats
  ])

  tensor = transform(image)
  tensor = tensor.unsqueeze(0)  # add batch dim -> [1, 1, 28, 28]
  return tensor


def predict(img_array: np.ndarray) -> str:
  tensor = _to_fashion_mnist(img_array)
  with torch.inference_mode():
    logits = torch_classifier_model(tensor)
    return CLASS_NAMES[logits.argmax(dim=1)]