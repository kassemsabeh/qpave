from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext
import torch

# Class to build vocabulary for the "original attribute" or "coarse grained attribute"
class Vocab():
    def __init__(self, attributes: list, config: dict) -> None:
        self.tokenizer = get_tokenizer('basic_english')
        self.attributes = attributes
        self.max_len = 6
        self.config = config
        self.vocab = build_vocab_from_iterator(self.yield_tokens(attributes), specials=["<unk>", "<pad>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        self.vocab_size = len(self.vocab)
        

    def yield_tokens(self, data):
        for text in data:
            out = self.tokenizer(text)
            # if len(out) > self.max_len:
            #     self.max_len = len(out)
            yield out
    
    def tokenize(self, inputs: list):
        # Define the text pipeline
        text_pipeline = lambda x: self.vocab(self.tokenizer(x)) 
        # Tokenize the sequence
        outputs = [text_pipeline(word) for word in inputs]
        outputs = [a + [1] * (self.max_len - len(a)) for a in outputs]
        return outputs

    def generate_embedding_matrix(self):
        self.glove = torchtext.vocab.GloVe(name='6B', dim=self.config['embeddings_dim'])
        weights_matrix = torch.zeros((len(self.vocab), self.config['embeddings_dim']))
        # weights_matrix = torch.zeros(238, self.config['embeddings_dim'])
        for index, word in enumerate(self.vocab.get_stoi()):
            try:
                weights_matrix[index] = self.glove[word]
            except:
                weights_matrix[index] = torch.random.normal(scale=0.6, size=(self.config['embeddings_dim'],))
        return weights_matrix
