# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import bs4
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace) -> tuple: # (dict, dict, dict, int)
    """
    Load data from huggingface datasets.
    If dataset is not in huggingface datasets, takes data from local directory.

    Args:
        dataset_name (str): Dataset name.
        args (argparse.Namespace): Arguments.
        train_valid_split (float): Train-valid split ratio.

    Returns:
        train_data (dict): Training data. (text, label)
        valid_data (dict): Validation data. (text, label)
        test_data (dict): Test data. (text, label)
        num_classes (int): Number of classes.
    """

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'label': []
    }
    valid_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'label': []
    }
    test_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'label': []
    }

    if name == 'sts_b':
        dataset = load_dataset('glue', 'stsb')

        train_df = pd.DataFrame(dataset['train'])
        valid_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 1

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['sentence1'].tolist()
        train_data['text2'] = train_df['sentence2'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['sentence1'].tolist()
        valid_data['text2'] = valid_df['sentence2'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['sentence1'].tolist()
        test_data['text2'] = test_df['sentence2'].tolist()
        test_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:
    """
    Main function for preprocessing.

    Args:
        args (argparse.Namespace): Arguments.
    """

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    # Define tokenizer & config
    model_name = get_huggingface_model_name(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'indices': train_data['idx'],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'valid': {
            'indices': valid_data['idx'],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'test': {
            'indices': test_data['idx'],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['text1'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            # Get text and label
            text1 = split_data['text1'][idx]
            text2 = split_data['text2'][idx]
            label = split_data['label'][idx]

            # Tokenize text
            tokenized = tokenizer(text1, text2, padding='max_length', truncation=True,
                                  max_length=args.max_seq_len, return_tensors='pt')

            # Append tokenized data to data_dict
            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: # roberta does not use token_type_ids
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.float32)) # MSE loss

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
