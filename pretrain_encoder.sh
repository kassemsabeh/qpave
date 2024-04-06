DATA_DIRECTORY="./data/mlm-dataset"
MODEL_SAVE_DIRECTORY="./saved_models"

python3 run_pretrain.py \
--model_checkpoint='bert-base-uncased' \
--num_epochs=2 \
--train_file_path="${DATA_DIRECTORY}/train.jsonl" \
--val_file_path="${DATA_DIRECTORY}/val.jsonl" \
--output_model_path="${MODEL_SAVE_DIRECTORY}" \
--output_model_name="${MODEL_SAVE_DIRECTORY}/pretrained_model"
