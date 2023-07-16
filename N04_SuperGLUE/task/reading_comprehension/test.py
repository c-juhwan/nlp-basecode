# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import json
import time
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import pandas as pd
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.multi_classification.model import ClassificationModel
from model.reading_comprehension.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def testing(args: argparse.Namespace) -> None:
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    if len(logger.handlers) > 0:
        logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.log_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading dataset")
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['valid'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'valid_processed.pkl'))
    dataset_dict['test'] = CustomDataset(os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type, f'test_processed.pkl'))

    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    dataloader_dict['test'] = DataLoader(dataset_dict['test'], batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_dict['valid'].vocab_size
    args.num_classes = dataset_dict['valid'].num_classes
    args.pad_token_id = dataset_dict['valid'].pad_token_id

    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_dict['test'])} / {len(dataloader_dict['test'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = ClassificationModel(args).to(device)

    # Load model weights
    write_log(logger, "Loading model weights")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset, args.model_type, 'final_model.pt')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    write_log(logger, f"Loaded model weights from {load_model_name}")
    del checkpoint

    # Load Wandb
    if args.use_wandb:
        import wandb
        from wandb import AlertLevel
        wandb.init(project=args.proj_name,
                   name=get_wandb_exp_name(args) + f' - Test',
                   config=args,
                   notes=args.description,
                   tags=["TEST",
                         f"Dataset: {args.task_dataset}",
                         f"Model: {args.model_type}"])

    # Test - Start testing on valid dataset
    model = model.eval()
    valid_acc_cls = 0
    valid_f1_cls = 0
    for valid_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc="Testing - Valid set", position=0, leave=True)):
        # Test - Get input data
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        labels = data_dicts['labels'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            classification_logits = model(input_ids, attention_mask, token_type_ids)

        # Test - Calculate accuracy/f1 score
        batch_acc_cls = (classification_logits.argmax(dim=-1) == labels).float().mean()
        batch_f1_cls = f1_score(labels.cpu().numpy(), classification_logits.argmax(dim=-1).cpu().numpy(), average='macro')

        # Valid - Logging
        valid_acc_cls += batch_acc_cls.item()
        valid_f1_cls += batch_f1_cls

    valid_acc_cls /= len(dataloader_dict['valid'])
    valid_f1_cls /= len(dataloader_dict['valid'])

    # Test - Start testing on test dataset
    model = model.eval()
    test_result_list = []

    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc="Testing - Test set", position=0, leave=True)):
        # Test - Get input data
        indices = data_dicts['indices']
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        entities = data_dicts['entities']

        # Test - Forward pass
        with torch.no_grad():
            classification_logits = model(input_ids, attention_mask, token_type_ids)
            classification_pred = classification_logits

        # Translate indices (dict of tensors) to list of dicts
        indices_ = []
        for i in range(len(indices['passage'])):
            indices_.append({
                'passage': indices['passage'][i].item(),
                'query': indices['query'][i].item()
            })
        indices = indices_

        # Test - Save prediction
        for idx, pred, entity in zip(indices, classification_pred, entities):
            # Translate idx to passage_idx, question_idx, answer_idx
            passage_idx = idx['passage']
            query_idx = idx['query']
            pred = pred.max(dim=-1)[0] # Get max confidence

            new_test_dict = {
                'idx': passage_idx,
                'query': {
                    'idx': query_idx,
                    'entity_pred': [
                        {
                            entity: pred.item()
                        }
                    ]
                }
            }

            if len(test_result_list) == 0:
                test_result_list.append(new_test_dict)
            else:
                if test_result_list[-1]['idx'] == passage_idx: # If there's already a passage with same passage_idx
                    if test_result_list[-1]['query']['idx'] == query_idx: # If there's already a query with same query_idx
                        test_result_list[-1]['query']['entity_pred'].append(new_test_dict['query']['entity_pred'][0])
                    else: # If there's no query with same query_idx
                        test_result_list.append(new_test_dict)
                else: # If there's no passage with same passage_idx and this is a new passage
                    test_result_list.append(new_test_dict)

    # Test - Pick the most confident prediction as the final prediction
    final_test_dict = {'idx': [], 'label': []}
    for each_test_result in test_result_list:
        final_test_dict['idx'].append(each_test_result['query']['idx']) # Save query_idx as idx
        #print(each_test_result)

        # Get the most confident prediction
        max_confidence = 0

        for each_entity_pred in each_test_result['query']['entity_pred']:
            for entity, confidence in each_entity_pred.items():
                if confidence > max_confidence:
                    max_confidence = confidence
                    max_entity = entity

        final_test_dict['label'].append(max_entity)

    # Test - Save test results to jsonl file
    test_save_path = os.path.join(args.result_path, args.task, args.task_dataset, args.model_type)
    check_path(test_save_path)
    test_save_name = os.path.join(test_save_path, 'test_results.jsonl')

    # Test - Sort test results
    final_test_dict['idx'], final_test_dict['label'] = zip(*sorted(zip(final_test_dict['idx'], final_test_dict['label'])))

    with open(test_save_name, 'w') as f:
        for idx, pred in zip(final_test_dict['idx'], final_test_dict['label']):
            f.write(json.dumps({"idx": idx, "label": pred}, ensure_ascii=False) + '\n')

    # Test - Log test results
    write_log(logger, f"Valid accuracy: {valid_acc_cls:.4f}")
    write_log(logger, f"Valid f1 score: {valid_f1_cls:.4f}")
    write_log(logger, f"Test results saved to {test_save_name}")

    if args.use_tensorboard:
        writer.add_scalar('TEST/VAL_Acc', valid_acc_cls, 0)
        writer.add_scalar('TEST/VAL_F1', valid_f1_cls, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'valid_acc_cls': [valid_acc_cls],
            'valid_f1_cls': [valid_f1_cls]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result_Valid': wandb_table})
        wandb.save(test_save_name)

        wandb.finish()
