DATASET=sst2
BS=32
LR=5e-5
EP=10

clear

MODEL=bert
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=transformer_enc
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=cnn
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=gru
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=lstm
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=rnn
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=roberta
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=albert
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=electra
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=deberta
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

MODEL=debertav3
python main.py --task=classification --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
python main.py --task=classification --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=classification --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}