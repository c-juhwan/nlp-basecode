MODEL=bert
BS=32
LR=5e-5
EP=5

clear

TASK=single_classification
DATASET=cola
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=128
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

DATASET=sst2
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=128
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=mrpc
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=qqp
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=qnli
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=rte
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=wnli
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
# python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=mnli_m
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=mnli_mm
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# # python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=multi_classification
DATASET=ax
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=256
# # python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

TASK=regression
DATASET=sts_b
# python main.py --task=${TASK} --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL} --max_seq_len=128
# python main.py --task=${TASK} --job=training --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP} --optimize_objective=loss
python main.py --task=${TASK} --job=testing --task_dataset=${DATASET} --model_type=${MODEL} --batch_size=${BS} --learning_rate=${LR} --num_epochs=${EP}

python main.py --task=submission --job=submission --model_type=${MODEL}