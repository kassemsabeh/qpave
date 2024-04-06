# Datasets

The repository contains two datasets:   AZ-dataset and mlm-dataset. The AZ-dataset is used for the fine grained attribute value extraction task. The mlm-dataset is used to pre-train the BERT encoder on a masked language modelling task (MLM) task.  Both datasets are derived from a public product collection: the [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html).

## Download
The datasets can be downloaded [here](https://www.dropbox.com/sh/x0z5mnz3fe98lo4/AAA_6VzJrDj6N5OKr3WJQbw5a?dl=0). Download and unzip the files in this directory. This directory should contain two folders: AZ-dataset and mlm-dataset:

```bash
├── data
│   ├── AZ-dataset
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   ├── mlm-dataset
│   │   ├── train.jsonl
│   │   ├── val.jsonl
```

## AZ Dataset
The dataset contains 380k fine grained attribute value annotations from 320k distinct products across 571 categories. It is a large and diverse dataset obtained from 9 product domains for fine grained product attribute vale extraction study.

We provide full human annotations for the test set for evaluation. The rest of the data are obtained through distant supervision.

The dataset is in [JSON Lines](https://jsonlines.org/) format, with the following schema:

```json
{
  "id": A unique identifier,
  "category": Category name,
  "original attribute": Coarse grained attribute name,
  "context": Coarse grained attribute description text,
  "question": Fine grained attribute name,
  "answers": [
    {
      "text": Target fine grained attribute value,
      "answer_start": The begin character level index of the attribute value in the context,
    }
  ]
}
```
The "id" is a unique identifier used to identify each JSON object. Each JSON object contains a tuple of tagged fine grained and coarse grained attributes. An example is shown as follows:

```json
{
  "id": 790,
  "category": Electric Shavers,
  "original attribute": Color,
  "context": Black Foil Razor,
  "question": Head Type,
  "answers": [
    {
      "text": Foil,
      "answer_start": 6
    }
  ]
}
```
The dataset contains the following files:

* `train.jsonl`: Training set obtained with distant supervision. The training set contains 80% of the dataset. It is used to train the model.

* `val.jsonl`: Validation set obtained with distant supervision. The validation set contains 10% of the dataset. It is used for optimal hyper-parameter selection.

* `test.jsonl`: Test set with full human annotations. This set is used for the final evaluation.

The following table shows the overall statistics of the AZ-dataset:

|                     **Statistic**                    | **Count** |
|:----------------------------------------------------:|:---------:|
|                      # products                      |   322254  |
|                # attribute-value pairs               |   382180  |
|                       # domains                      |     9     |
|                  # unique categories                 |    571    |
|           # unique fine grained attributes           |    175    |
|          # unique coarse grained attributes          |    207    |
|                    # unique values                   |   141712  |
|    # unique category-fine grained attribute pairs    |    7745   |
|   # unique category-coarse grained attribute pairs   |    7664   |
| # unique fine grained-coarse grained attribute pairs |    3940   |

## MLM Dataset
The dataset contains over 8 million product textual descriptions obtained from over 1300 product categories. This dataset is derived from the [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/index.html). The dataset is used in an unsupervised fashion to pre-train the contextual encoder using a Masked Language Modeling (MLM) task. 

The dataset is in [JSON Lines](https://jsonlines.org/) format, with the following schema:

```json
{
  "id": A unique identifier,
  "text": Paragraph text
}
```
The dataset contains two files:

* `train.jsonl`: Training set containing 80% of the dataset and used to train the model.

* `val.jsonl`: Validation set containing 20% of the dataset and used for hyper-parameter tuning.

## Reading the Data in DataFrames

A simple script to read any of the [JSON Lines](https://jsonlines.org/) data in `data/` directory as [pandas](https://pandas.pydata.org/) dataframe:

```
import json
import pandas as pd

def getDF(path):
   with open(path) as f:
      i = 0
      df = {}
      for line in f:
         df[i] = json.loads(line)
         i += 1
   return pd.DataFrame.from_dict(df, orient='index')
   
df = getDF('data/AZ-dataset/train.jsonl')

```
