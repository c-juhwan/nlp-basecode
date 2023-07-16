MODEL=bert
BS=32
LR=5e-5
EP=5

clear

# TASK=multiple_choice
# DATASET=copa
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=512
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=cuda:2
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=reading_comprehension
DATASET=record
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=512
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=3 --device=cuda:0
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=3 --device=cuda:0

python main.py --task=submission --job=submission