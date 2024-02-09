DATASET=wmt16_en_de
BS=32
LR=5e-5
EP=10
MODEL=lstm

clear

python main.py --task=translation --job=preprocessing --task_dataset=${DATASET} --model_type=${MODEL}