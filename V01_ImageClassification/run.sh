DATASET=mnist
BS=32
LR=5e-5
EP=10
DEVICE=cuda
clear

MODEL=resnet50
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}

DATASET=fashion_mnist
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}

DATASET=cifar10
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --device=${DEVICE}
