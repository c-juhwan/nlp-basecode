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
        'text3': [],
        'label': []
    }
    valid_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'text3': [],
        'label': []
    }
    test_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'text3': [],
        'label': []
    }

    if name in ['boolq', 'boolq_few']:
        dataset_full = load_dataset('super_glue', 'boolq')
        dataset_few = load_dataset('juny116/few_glue', 'boolq')

        if name == 'boolq':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'boolq_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['question'].tolist()
        train_data['text2'] = train_df['passage'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['question'].tolist()
        valid_data['text2'] = valid_df['passage'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['question'].tolist()
        test_data['text2'] = test_df['passage'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name in ['cb', 'cb_few']:
        dataset_full = load_dataset('super_glue', 'cb')
        dataset_few = load_dataset('juny116/few_glue', 'cb')

        if name == 'cb':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'cb_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 3

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['premise'].tolist()
        train_data['text2'] = train_df['hypothesis'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['premise'].tolist()
        valid_data['text2'] = valid_df['hypothesis'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['premise'].tolist()
        test_data['text2'] = test_df['hypothesis'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name in ['multirc', 'multirc_few']:
        dataset_full = load_dataset('super_glue', 'multirc')
        dataset_few = load_dataset('juny116/few_glue', 'multirc')

        if name == 'multirc':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'multirc_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['paragraph'].tolist()
        train_data['text2'] = train_df['question'].tolist()
        train_data['text3'] = train_df['answer'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['paragraph'].tolist()
        valid_data['text2'] = valid_df['question'].tolist()
        valid_data['text3'] = valid_df['answer'].tolist()
        valid_data['label'] = valid_df['label'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['paragraph'].tolist()
        test_data['text2'] = test_df['question'].tolist()
        test_data['text3'] = test_df['answer'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name in ['rte', 'rte_few']:
        dataset_full = load_dataset('super_glue', 'rte')
        dataset_few = load_dataset('juny116/few_glue', 'rte')

        if name == 'rte':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'rte_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['premise'].tolist()
        train_data['text2'] = train_df['hypothesis'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['premise'].tolist()
        valid_data['text2'] = valid_df['hypothesis'].tolist()
        valid_data['label'] = valid_df['label'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['premise'].tolist()
        test_data['text2'] = test_df['hypothesis'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'axb':
        dataset = load_dataset('super_glue', 'axb')

        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['sentence1'].tolist()
        test_data['text2'] = test_df['sentence2'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'axg':
        dataset = load_dataset('super_glue', 'axg')

        test_df = pd.DataFrame(dataset['test'])
        num_classes = 2

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['premise'].tolist()
        test_data['text2'] = test_df['hypothesis'].tolist()
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
            if args.task_dataset in ['multirc', 'multirc_few']:
                text3 = split_data['text3'][idx]
            label = split_data['label'][idx]

            # Tokenize text
            if args.task_dataset in ['boolq', 'boolq_few', 'cb', 'cb_few', 'rte', 'rte_few', 'axb', 'axg']:
                tokenized = tokenizer(text1, text2, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')
            elif args.task_dataset in ['multirc', 'multirc_few']:
                text1 = text1 + ' ' + text2 # Paragraph + Question
                tokenized = tokenizer(text1, text3, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')

            # Append tokenized data to data_dict
            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: # roberta does not use token_type_ids
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
