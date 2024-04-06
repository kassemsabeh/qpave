# QPAVE: A Multi-task Question Answering Approach for Fine-Grained Product Attribute Value Extraction

This repository contains the datasets and source code used in our paper: QPAVE: A Multitask Question Answering Approach for Fine-Grained Product Attribute Value Extraction.

## Installation
1. Clone the repository
2. Download the datasets [**HERE**](https://anonymous.4open.science/r/QPAVE/data/README.md). Once you unzip the downloaded files, please put the downloaded folders under the directory ```./data/```.
3. Install the required dependencies in the ```requirements.txt``` file:
    ```
    $ pip install -r requirements.txt
    ```

## MLM Pre-training
The default encoder used is ```BERT-base-uncased``` from [Hugging Face](https://huggingface.co/bert-base-uncased). To pre-train the encoder model on the Masked Language Modelling (MLM) task using the ```./data/mlm-dataset```, run the following shell script:
```
$ bash ./pretrain_encoder.sh
```

The resulting pre-trained model will be stored in ```./saved_models/pretrained_model```.
> **_NOTE:_**  You can use any model in the 🤗 Transformers library by changing the ```model_checkpoint``` in the ```pretrain_encoder.sh``` script.

## Model Training
To train the model for the fine grained attribute value extraction task using the ```./data/AZ-dataset```, run the following shell script:

```
$ bash ./train_model.sh
```
The model uses the pre-trained encoder by default to train the model. The trained model will be stored in ```./saved_models/qpave.pt```.

After running all scripts, you should obtain the following directory tree:
```bash
├── README.md
├── config.py
├── data
│   └── AZ-dataset
│   └── mlm-dataset
├── saved_models
│   └── pretrained_model
│   └── qpave.pt
├── data_manager.py
├── models.py
├── pretrain_encoder.sh
├── requirements.txt
├── run.py
├── run_pretrain.py
├── train_model.sh
├── trainer.py
├── utils.py
└── vocabulary.py
```
## Model Hyper-parameters
For reproducability, we provide all hyper-parameter details of the models used in the paper:

|            **Hyper-parameters**           |   **value**  |
|:-------------------------------------:|:-----------------:|
|                 **Common**            |                   |
|               batch size              |         64        |
|               optimizer               |        Adam       |
|         learning rate schedule        |    linear decay   |
|         initial learning rate         |     $1e^{-5} $    |
|          max sequence length          |        384        |
|            overflow stride            |        128        |
|          **BERT-MLC**                 |                   |
|          number classes               |      122176       |
|          word embedding size          |        768        |
|          pre-trained checkpoint       | BERT-base-uncased |
|          **SUOpenTag**                |                   |
|          word embedding size          |        768        |
|               LSTM units              |        256        |
|         LSTM recurrent dropout        |        0.3        |
|            attention heads            |         4         |
|           attention dropout           |        0.0        |
|       word embedding layer init       | BERT-base-uncased |
|                 **AVEQA**             |                   |
|        hidden dimension ($d_h$)       |        768        |
|         pre-trained checkpoint        | BERT-base-uncased |
|                 **MAVEQA**            |                   |
|        max long sequence length       |        1024       |
|      max global sequence length       |       64          |
|         pre-trained checkpoint        |[ETC-base-2x-pretrain](https://github.com/google-research/google-research/tree/master/etcmodel#pre-trained-models)         |
|                  **QPAVE**            |                   |
|           lambda ($\lambda$)          |        0.8        |
|   hidden embedding dimension ($d_h$)  |        768        |
| attribute embedding dimension ($d_r$) |        100        |
|     attribute embedding vocab size    |        238        |
|        attribute embedding init       |       GloVe       |
|          number of categories         |        572        |
|         pre-trained checkpoint        | BERT-base-uncased |



[datasets]: /data/README.md
