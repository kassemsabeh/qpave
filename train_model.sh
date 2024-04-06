DATA_DIRECTORY="./data/AZ-dataset/"
MODEL_SAVE_DIRECTORY="./saved_models"

python3 run.py \
--pretrained_encoder_checkpoint="./saved_models/pretrained_model" \
--train_file_path="${DATA_DIRECTORY}/train.jsonl" \
--val_file_path="${DATA_DIRECTORY}/val.jsonl" \
--test_file_path="${DATA_DIRECTORY}/test.jsonl" \
--output_model_path="${MODEL_SAVE_DIRECTORY}" \
--output_model_name="${MODEL_SAVE_DIRECTORY}/qpave.pt"
