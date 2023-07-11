# Standard Library Modules
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # This prevents tokenizers from taking all cpus
import sys
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
# Pytorch Modules
import torch
torch.set_num_threads(2) # This prevents Pytorch from taking all cpus
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.regression.model import RegressionModel
from model.regression.dataset import CustomDataset
from utils.utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_wandb_exp_name, get_torch_device, check_path

def testing(args: argparse.Namespace) -> tuple: # (test_acc_cls, test_f1_cls)
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
    model = RegressionModel(args).to(device)

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
    valid_result = {'pred': [], 'true': []}
    for valid_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc="Testing - Valid set", position=0, leave=True)):
        # Test - Get input data
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)
        labels = data_dicts['labels'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            regression_pred = model(input_ids, attention_mask, token_type_ids)

        # Test - Unbatch & Save to result dictionary
        for pred, true in zip(regression_pred, labels):
            valid_result['pred'].append(pred.item())
            valid_result['true'].append(true.item())

    # Test - Calculate metrics for valid dataset
    # Use Pearson & Spearman correlation coefficient
    valid_pearsonr = np.round(pearsonr(valid_result['pred'], valid_result['true'])[0], 4)
    valid_spearmanr = np.round(spearmanr(valid_result['pred'], valid_result['true'])[0], 4)

    # Test - Start testing on test dataset
    model = model.eval()
    test_save_dict = {'index': [], 'prediction': []}
    for test_iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['test'], total=len(dataloader_dict['test']), desc="Testing - Test set", position=0, leave=True)):
        # Test - Get input data
        indices = data_dicts['indices']
        input_ids = data_dicts['input_ids'].to(device)
        attention_mask = data_dicts['attention_mask'].to(device)
        token_type_ids = data_dicts['token_type_ids'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            regression_pred = model(input_ids, attention_mask, token_type_ids)

        # Test - Unbatch & Save
        for idx, pred in zip(indices, regression_pred):
            test_save_dict['index'].append(idx.item())
            # Bound prediction value to 0 ~ 5
            pred = torch.clamp(pred, 0, 5)
            test_save_dict['prediction'].append(np.round(pred.item(), 3))

    # Test - Save test result to tsv file
    test_save_df = pd.DataFrame(test_save_dict)
    test_save_df = test_save_df.sort_values(by=['index'])
    check_path(os.path.join(args.result_path, args.task, args.task_dataset, args.model_type))
    test_save_df.to_csv(os.path.join(args.result_path, args.task, args.task_dataset, args.model_type, 'test_result.tsv'), sep='\t', index=False)

    # Test - Log test results
    write_log(logger, f"Valid Pearson correlation coefficient: {valid_pearsonr}")
    write_log(logger, f"Valid Spearman correlation coefficient: {valid_spearmanr}")
    write_log(logger, f"Test results saved to {os.path.join(args.result_path, args.task, args.task_dataset, args.model_type, 'test_result.tsv')}")

    if args.use_tensorboard:
        writer.add_scalar('TEST/VAL_Pearson', valid_pearsonr, 0)
        writer.add_scalar('TEST/VAL_Spearman', valid_spearmanr, 0)
        writer.close()
    if args.use_wandb:
        wandb_df = pd.DataFrame({
            'valid_pearson_reg': [valid_pearsonr],
            'valid_spearman_reg': [valid_spearmanr]
        })
        wandb_table = wandb.Table(dataframe=wandb_df)
        wandb.log({'TEST_Result_Valid': wandb_table})
        wandb.save(os.path.join(args.result_path, args.task, args.task_dataset, args.model_type, 'test_result.tsv'))

        wandb.finish()
