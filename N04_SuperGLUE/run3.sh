MODEL=bert
BS=32
LR=5e-5
EP=5

clear

TASK=span_classification
DATASET=wic
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=128
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=cuda:0
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=cuda:0

TASK=span_classification
DATASET=wsc
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=128
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=cuda:0
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=cuda:0