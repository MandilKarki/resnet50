import os
from skimage import io

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class CustomDataset(Dataset):
  def __init__(self, root_dir, transform=None, is_test_set=False):
    self.root_dir = root_dir
    self.transform = transform
    self.is_test_set = is_test_set
    self.data = ImageFolder(root_dir, transform=transform)
    self.classes = self.data.classes
 
  def __getitem__(self, index):
    image, label = self.data[index]

    if self.is_test_set:
      return image
    else:
      return(image, label)
 
  def __len__(self):
    return len(self.data)

  def get_num_classes(self):
    return len(self.classes)