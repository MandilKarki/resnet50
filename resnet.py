import torch.nn as nn
from torchvision import datasets, models, transforms

class Resnet(nn.Module):
  def __init__(self, model_name, input_channels, num_classes,
               load_pretrained_weights=True, train_only_last_layer=False):
    super(Resnet, self).__init__()
    self.model_name = model_name
    self.input_channels = input_channels
    self.num_classes = num_classes
    self.load_pretrained_weights = load_pretrained_weights
    self.train_only_last_layer = train_only_last_layer
    #self.flatten = nn.layers.Flatten()
    if self.load_pretrained_weights:
      self.features = models.resnet50(pretrained=True)
    else:
      self.features = models.resnet50(pretrained = False)
    if self.train_only_last_layer:
      for param in self.features.parameters():
        param.requires_grad = False
    in_ftrs = self.features.fc.in_features
    self.features.fc = nn.Linear(in_ftrs, self.num_classes)
    self.classifier_layer = nn.Sequential(
            nn.Linear(in_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512 , self.num_classes)
            # nn.Linear(256 , self.num_classes)
        )

  def forward(self, inputs):
    x = self.features(inputs)
    x = self.classifier_layer(inputs)
    return x
