from data_manager import ValDataset

import collections
import itertools

import transformers
import datasets
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from sklearn import metrics


# Function to load the dataset in DatasetDict format for train, val, and test
def load_dataset(config: dict) -> DatasetDict:
    train = Dataset.from_json(config['train_file_path'])
    validation = Dataset.from_json(config['validation_file_path'])
    test = Dataset.from_json(config['test_file_path'])
    dataset = DatasetDict()
    dataset['train'] = train
    dataset['validation'] = validation
    dataset['test'] = test
    return dataset

# Function to group test datasets by groups if exists for attribute training
def create_test_dataset(config: dict) -> DatasetDict:
    print('Reading test datasets...')
    my_list = []
    for cat in config['categories']:
        print(f"Reading {cat} category from local files")
        file_path = f"few_shot_data/test/{'_'.join(cat.split())}.jsonl"
        data = Dataset.from_json(file_path)
        my_list.append(data)
    dataset = DatasetDict()
    for cat, data in zip(config['categories'], my_list):
        dataset[cat] = data
    return dataset

# Create a dictionary that maps the encodings of the categories for creating the classification training data
def create_encoding_dict(data: list) -> dict:
  list_categories = []
  list_attributes = []
  for value in data:
    list_categories.append(value['category'])
    list_attributes.append(value['original_attribute'].lower())
  set_categories = list(set(list_categories))
  cat2idx = {category: i for i, category in enumerate(set_categories, 1)}
  cat2idx['unk'] = 0
  idx2cat = {i: category for category, i in cat2idx.items()}
  return cat2idx, idx2cat, list(set(list_attributes))

# Function to post process predictions of the model
def postprocess_qa_predictions(examples, tokenizer, features, all_start_logits, all_end_logits, n_best_size = 20, max_answer_length = 30):
    negative_answers = False
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    try:
                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        # -> Added condition here
                        if context[start_char: end_char] == context:
                            continue
                    except:
                        continue
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}
        
        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        if not negative_answers:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
            predictions[example["id"]] = answer

    return predictions

# Function to produce predictions given the dataset and the model   
def predict(validation_dataset, model, device, config):
  start = []
  end = []
  model.to(device)
  model.eval()
  loader_parameters = {
      'batch_size': config['batch_size'],
      'shuffle': False,
      'num_workers': 0
  }
  val_loader = DataLoader(validation_dataset, **loader_parameters)
  for _, data in enumerate(val_loader):
    ids = torch.tensor(data['ids']).to(device, dtype=torch.long)
    mask = torch.tensor(data['mask']).to(device, dtype=torch.long)
    original_attribute = data['original_attributes'].to(device, dtype=torch.long)
    predicted_start, predicted_end, predicted_category = model(ids, mask, original_attribute)
    start.append(predicted_start.cpu().detach().numpy().tolist())
    end.append(predicted_end.cpu().detach().numpy().tolist())
  start = list(itertools.chain.from_iterable(start))
  end = list(itertools.chain.from_iterable(end))
  return start, end

# Function to calculate the metrics after the predictions for the model
def calculate_metrics(references, predictions):
    truth = [val['answers']['text'][0] for val in references]
    predicted = [val['prediction_text'] for val in predictions]
    wrong_indices = np.where(np.array(truth) != np.array(predicted))[0]
    accuracy = metrics.accuracy_score(truth, predicted)
    my_metrics = {
        # f"{category}_accuracy": metrics.accuracy_score(truth, predicted),
        f"accuracy": accuracy

    }
    return my_metrics, list(wrong_indices), predicted
