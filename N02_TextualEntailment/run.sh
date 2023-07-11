DATASET=snli
BS=32
LR=5e-5
EP=5

clear

MODEL=bert
python main.py --task=entailment --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=entailment --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=entailment --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=albert
python main.py --task=entailment --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=entailment --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=entailment --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=electra
python main.py --task=entailment --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=entailment --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=entailment --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=deberta
python main.py --task=entailment --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=entailment --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=entailment --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=debertav3
python main.py --task=entailment --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=entailment --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=entailment --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
