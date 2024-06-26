config = dict(
    epochs = 10,
    batch_size = 32,
    negative = False,
    stride = 128,
    max_len = 384,
    learning_rate = 1e-05,
    log_every = 500,
    model_checkpoint = 'bert-base-uncased',
    dataset='tgap-dataset',
    file_path = 'dataset/validation',
    train_file_path='data/train.jsonl',
    validation_file_path='data/val.jsonl',
    test_file_path='data/test.jsonl',
    alpha = 0.4,
    save_model = True,
    model_save_path = 'saved_models/egap_checkpoint.pt',
    use_categories=True,
    track_errors = False,
    embeddings_dim = 100
)
