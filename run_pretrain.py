import os
from absl import flags
import sys

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

_MODEL_CHECKPOINT = flags.DEFINE_string(
    'model_checkpoint',
    default=None,
    help='BERT model checkpoint',
    required=True
)

_NUM_EPOCHS = flags.DEFINE_string(
    'num_epochs',
    default=None,
    help='Number of training epochs for the model',
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
os.environ["WANDB_DISABLED"] = "true"
FLAGS = flags.FLAGS
FLAGS(sys.argv)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(_MODEL_CHECKPOINT.value, use_fast=True)
# Set block size
block_size = tokenizer.model_max_length

#Function to load dataset
def load_dataset(train_path, val_path):
    train = Dataset.from_json(train_path)
    val = Dataset.from_json(val_path)
    dataset = DatasetDict()
    dataset['train'] = train
    dataset['validation'] = val
    return dataset

# Function to group texts
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def tokenize_function(examples):
    return tokenizer(examples["text"])

def main():
    # Create output directory if it does not exist
    if not os.path.exists(_OUTPUT_MODEL_PATH.value):
        os.mkdir(_OUTPUT_MODEL_PATH.value)
    # Load the dataset
    datasets = load_dataset(_INPUT_TRAIN_PATH.value, _INPUT_VAL_PATH.value)

    # Tokenize the dataset
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # Group texts and chunk them
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=4)

    # Define the model
    model = AutoModelForMaskedLM.from_pretrained(_MODEL_CHECKPOINT.value)

    # Define training arguments
    model_name = _MODEL_CHECKPOINT.value.split("/")[-1]
    training_args = TrainingArguments(f"{model_name}-finetuned-egap", evaluation_strategy = "epoch", save_strategy='no', learning_rate=2e-5, weight_decay=0.01, per_device_train_batch_size=32,
                                       per_device_eval_batch_size=32, num_train_epochs=int(_NUM_EPOCHS.value), push_to_hub=False, load_best_model_at_end=False)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=lm_datasets["train"], eval_dataset=lm_datasets["validation"], 
                      data_collator=data_collator)
    
    # Start training
    trainer.train()

    # Save model
    trainer.save_model(_OUTPUT_MODEL_NAME.value)

if __name__ == '__main__':
    main()
