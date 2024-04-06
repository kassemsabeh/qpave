import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# Define Hypernetwork model
class HyperNetwork(nn.Module):
  def __init__(self, vocab_size, num_weights, weights_matrix, embed_dim):
    super(HyperNetwork, self).__init__()
    # self.embedding = nn.Embedding(vocab_size, embed_dim)
    # Set freeze to false
    # This is to use glove pretrained embeddings
    self.embedding = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx=1)
    self.relu = nn.LeakyReLU()
    self.linear = nn.Linear(embed_dim, num_weights)
    self.activation = nn.Sigmoid()
  
  def forward(self, inputs):
    x = self.embedding(inputs)[0]
    x = self.relu(x)
    x = self.linear(x)
    x = self.activation(x)
    return x

# Implementation as in Ha et al.
class GeneralHyperNetwork(nn.Module):
    def __init__(self, k_size = 8, z_dim = 32, output_size=16, input_size=16):
        super(GeneralHyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.k_size = k_size
        self.output_size = output_size
        self.input_size = input_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.output_size*self.k_size*self.k_size)).cuda(),2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.output_size*self.k_size*self.k_size)).cuda(),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.input_size*self.z_dim)).cuda(),2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.input_size*self.z_dim)).cuda(),2))

    def forward(self, x):

        states = torch.matmul(x, self.w2) + self.b2
        states = states.view(self.input_size, self.z_dim)

        y = torch.matmul(states, self.w1) + self.b1
        kernel = y.view(self.output_size, self.input_size, self.k_size, self.k_size)

        return kernel

# Define the main QA model
class QAModel(nn.Module):
  def __init__(self, config, weights_matrix):
    super(QAModel, self).__init__()
    self.config = config
    self.model_config = AutoConfig.from_pretrained(self.config['model_checkpoint'], output_hidden_states=True)
    self.embeddings = AutoModel.from_pretrained(self.config['model_checkpoint'], config=self.model_config)
    self.qa_outputs = nn.Linear(768, 2, bias=False)
    self.dropout = nn.Dropout(0.3)
    self.hypernetwork = HyperNetwork(self.config['vocab_size'], 256, weights_matrix, self.config['embeddings_dim'])
    if self.config['use_categories']:
      self.classifier = nn.Linear(768, self.config['num_classes'])
      self.linear = nn.Linear(768, 768)
  
  def forward(self, input_ids, attention_mask, tokens):
    output_sequence = self.embeddings(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
    self.w = self.hypernetwork(tokens)
    self.w = self.w.reshape(2, 768)
    logits = F.linear(output_sequence, self.w)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)
    if self.config['use_categories']:
      cls = output_sequence[:, 0]
      cls = self.linear(cls)
      cls = torch.nn.ReLU()(cls)
      cls = self.dropout(cls)
      classification_output = self.classifier(cls)
      return start_logits, end_logits, classification_output
    else:
      return start_logits, end_logits, None
