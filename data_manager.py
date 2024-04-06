import torch
from torch.utils.data import Dataset

# Define the dataset class for the training set that is used to create the batches used to feed the pytorch model
class QADataset(Dataset):
  def __init__(self, data, cat2idx):
    self.len = len(data)
    self.data = data
    self.cat2idx = cat2idx

  def __getitem__(self, index):
    inputs = self.data[index]
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    start_positions = inputs['start_positions']
    end_positions = inputs['end_positions']
    original_attributes = inputs['original_attribute']
    try:
      target = self.cat2idx[self.data[index]['category']]
    except:
      target = self.cat2idx['unk']
    
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "start_pos": torch.tensor(start_positions, dtype=torch.long),
        "end_pos": torch.tensor(end_positions, dtype=torch.long),
        "targets": torch.tensor(target, dtype=torch.long),
        "original_attributes": torch.tensor(original_attributes, dtype=torch.long)
      }
      
  def __len__(self):
    return self.len

# Define the dataset class for the val/test set that is used to create the batches used to feed the pytorch model
class ValDataset(Dataset):
  def __init__(self, data, cat2idx):
    self.len = len(data)
    self.data = data
    self.cat2idx = cat2idx

  def __getitem__(self, index):
    inputs = self.data[index]
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    original_attributes = inputs['original_attribute']
    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "mask": torch.tensor(mask, dtype=torch.long),
        "original_attributes": torch.tensor(original_attributes, dtype=torch.long)
      }
      
  def __len__(self):
    return self.len
