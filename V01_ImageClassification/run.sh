DATASET=mnist
BS=32
LR=5e-5
EP=10

clear

MODEL=resnet50
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}