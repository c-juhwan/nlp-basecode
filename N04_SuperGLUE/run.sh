MODEL=bert
BS=32
LR=5e-5
EP=5

clear

# TASK=multi_classification
# DATASET=boolq
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=512
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# TASK=multi_classification
# DATASET=cb
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# TASK=multi_classification
# DATASET=multirc
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=512
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# TASK=multi_classification
# DATASET=rte
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# TASK=multi_classification
# DATASET=axb
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# # python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# TASK=multi_classification
# DATASET=axg
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# # python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

# python main.py --task=submission --job=submission --model_type=${MODEL}