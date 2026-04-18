import torch
from torch import nn

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

MODEL_PATH = '/torch_state/fashion_model_classifier_tiny_vgg_v2.pth'
torch_classifier_model = FashionModelV2(input_shape=1, hidden_units=10, output_shape=10)
torch_classifier_model.load_state_dict(torch.load(f=MODEL_PATH))
torch_classifier_model.eval()


def predict(x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = torch_classifier_model(x)
        return logits.argmax(dim=1)


'''Usage:                                                                                                                                                                                                                           
  from deep_learning.product_type_classifier import predict                                                                                                                                                                                                                                                                                                                                                                                                    
  # x should be a tensor of shape (batch, 1, 28, 28)                                                                                                                                                                               
  predictions = predict(x)      
  # it returns class index
'''