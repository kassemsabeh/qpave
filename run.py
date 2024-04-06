from config import config
from utils import *
from data_manager import QADataset
from vocabulary import Vocab
from models import QAModel
from trainer import QATrainer

import os
from absl import flags
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.auto.tokenization_auto import AutoTokenizer


_PRETRAINED_ENCODER_CHECKPOINT = flags.DEFINE_string(
    'pretrained_encoder_checkpoint',
    default=None,
    help='The path for the pretrained model',
    required=True
)

_INPUT_TRAIN_PATH = flags.DEFINE_string(
    'train_file_path',
    default=None,
    help='The input directory for the train jsonl file',
    required=True
)

_INPUT_VAL_PATH = flags.DEFINE_string(
    'val_file_path',
    default=None,
    help='The input directory for the val jsonl file',
    required=True
)

_INPUT_TEST_PATH = flags.DEFINE_string(
    'test_file_path',
    default=None,
    help='The input directory for the test jsonl file',
    required=True
)

_OUTPUT_MODEL_PATH = flags.DEFINE_string(
    'output_model_path',
    default=None,
    help='The output directory for saving the model',
    required=True
)

_OUTPUT_MODEL_NAME = flags.DEFINE_string(
    'output_model_name',
    default=None,
    help='The output name for saving the model',
    required=True
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

config['train_file_path'] = _INPUT_TRAIN_PATH.value
config['validation_file_path'] = _INPUT_VAL_PATH.value
config['test_file_path'] = _INPUT_TEST_PATH.value
config['model_save_path'] = _OUTPUT_MODEL_NAME.value

# Check if pretrained model exists
if os.path.exists(_PRETRAINED_ENCODER_CHECKPOINT.value):
    config['model_checkpoint'] = _PRETRAINED_ENCODER_CHECKPOINT.value

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = load_dataset(config)
tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'], use_fast=True)
cat2idx, idx2cat, list_attributes = create_encoding_dict(data['train'])
vocab = Vocab(list_attributes, config)
weights_matrix = vocab.generate_embedding_matrix()
config['vocab_size'] = vocab.vocab_size
config['num_classes'] = len(cat2idx)


# Function to extract the features out of the training data to feed the model
def prepare_train_features(examples):
    # Specify the padding side of the model
    pad_on_right = tokenizer.padding_side == 'right'
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=config['max_len'],
        stride=config['stride'],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    categories = []
    original_attributes = []
    
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        category = examples['category'][sample_index]
        original_attribute = examples['original_attribute'][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
        categories.append(category)
        original_attributes.append(original_attribute.lower())

    if config['use_categories']:
        tokenized_examples['category'] = categories

    tokenized_examples['original_attribute'] = vocab.tokenize(original_attributes)

    return tokenized_examples


# Function to extract the features from validation dataset for testing the model
def prepare_validation_features(examples):
  # Specify the padding side of the model
  pad_on_right = tokenizer.padding_side == 'right'
  # Some of the questions have lots of whitespace on the left, which is not useful and will make the
  # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
  # left whitespace
  examples["question"] = [q.lstrip() for q in examples["question"]]

  # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
  # in one example possible giving several features when a context is long, each of those features having a
  # context that overlaps a bit the context of the previous feature.
  tokenized_examples = tokenizer(
      examples["question" if pad_on_right else "context"],
      examples["context" if pad_on_right else "question"],
      truncation="only_second" if pad_on_right else "only_first",
      max_length=config['max_len'],
      stride=config['stride'],
      return_overflowing_tokens=True,
      return_offsets_mapping=True,
      padding="max_length",
  )
  categories = []
  original_attributes = []
  
  # Since one example might give us several features if it has a long context, we need a map from a feature to
  # its corresponding example. This key gives us just that.
  sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

  # We keep the example_id that gave us this feature and we will store the offset mappings.
  tokenized_examples["example_id"] = []

  for i in range(len(tokenized_examples["input_ids"])):
      # Grab the sequence corresponding to that example (to know what is the context and what is the question).
      sequence_ids = tokenized_examples.sequence_ids(i)
      context_index = 1 if pad_on_right else 0

      # One example can give several spans, this is the index of the example containing this span of text.
      sample_index = sample_mapping[i]
      tokenized_examples["example_id"].append(examples["id"][sample_index])
      category = examples['category'][sample_index]
      original_attribute = examples['original_attribute'][sample_index]

      # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
      # position is part of the context or not.
      tokenized_examples["offset_mapping"][i] = [
          (o if sequence_ids[k] == context_index else None)
          for k, o in enumerate(tokenized_examples["offset_mapping"][i])
      ]
      categories.append(category)
      original_attributes.append(original_attribute)
  if config['use_categories']:
    tokenized_examples['category'] = categories
  tokenized_examples['original_attribute'] = vocab.tokenize(original_attributes)
  return tokenized_examples


# Function to track errors in the predictikon of the model
def track_errors(indices, predictions, data):
    predictions = [predictions[i] for i in indices]
    ids = data[indices]['id']
    contexts = data[indices]['context']
    questions = data[indices]['question']
    answers = [answer['text'][0] for answer in data[indices]['answers']]
    categories = data[indices]['category']
    if config['track_errors']:
        df = pd.DataFrame(list(zip(ids, categories, contexts, questions, answers, predictions)),
                columns =['id', 'category', 'context', 'question', 'answer', 'predicted answer'])
        df.to_excel(f"errors.xlsx", index=False)
        return df
    return None

# Function to test the model on the data
def evaluate_model(model, data):
    validation_features = data.map(prepare_validation_features, batched=True,
                                                 remove_columns=data.column_names)
    validation_dataset = ValDataset(validation_features, cat2idx)
    predicted_start, predicted_end = predict(validation_dataset, model, device, config)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    final_predictions = postprocess_qa_predictions(data, tokenizer, validation_features, predicted_start, predicted_end)
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in data]
    my_metrics, wrong_indices, predictions = calculate_metrics(references, formatted_predictions)
    df = track_errors(wrong_indices, predictions, data)
    return my_metrics, formatted_predictions, references

def make(config):
  # Tokenize and prepare the data
  tokenized_data = data.map(prepare_train_features, batched=True, remove_columns=data['train'].column_names)
  # Declare the parameters used for the data loader class
  loader_parameters = {
      'batch_size': config['batch_size'],
      'shuffle': True,
      'num_workers': 0
  }

  # Instantiate a data loader class to feed into the dataset
  training_set = QADataset(tokenized_data['train'], cat2idx)
  validation_set = QADataset(tokenized_data['validation'], cat2idx)

  # Feed the dataset into the dataloader class before the training
  training_loader = DataLoader(training_set, **loader_parameters)
  validation_loader = DataLoader(validation_set, **loader_parameters)

  # Make the model
  # Instantiate a QA model
  model = QAModel(config, weights_matrix)
  model.to(device)
  
  # Define the loss and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])
  return model, training_loader, validation_loader, loss_function, optimizer

# Define the overall pipeline for model training
def model_pipeline(hyperparameters):
    model, train_loader, validation_loader, loss_function, optimizer = make(config)
    # print(model)
    trainer = QATrainer(train_loader, validation_loader, model, optimizer, loss_function, config, device)
    trainer.train()
    model = QAModel(config, weights_matrix)
    model.to(device)
    try:
        print("Loading saved model from local repository for evaluation......")
        model.load_state_dict(torch.load(config['model_save_path']))
    except:
        print("No saved model..loading final instance of trained model")
        model = trainer.model
    model.eval()
    metrics, predicted, truth = evaluate_model(model, data['test'])
    print(metrics)
    return trainer



def main():
    if not os.path.exists(_OUTPUT_MODEL_PATH.value):
        os.mkdir(_OUTPUT_MODEL_PATH.value)
    
    # Build, train and analyze the model with the pipeline
    trainer = model_pipeline(config)

if __name__ == '__main__':
    main()
