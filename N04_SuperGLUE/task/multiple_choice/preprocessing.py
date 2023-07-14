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
        'text4': [],
        'label': []
    }
    valid_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'text3': [],
        'text4': [],
        'label': []
    }
    text_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'text3': [],
        'text4': [],
        'label': []
    }

    if name in ['copa', 'copa_few']:
        dataset_full = load_dataset('super_glue', 'copa')
        dataset_few = load_dataset('juny116/few_glue', 'copa')

        if name == 'copa':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'copa_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['premise'].tolist()
        train_data['text2'] = train_df['choice1'].tolist()
        train_data['text3'] = train_df['choice2'].tolist()
        train_data['text4'] = []
        for i in range(len(train_df)):
            if train_df['question'][i] == "cause":
                train_data['text4'].append("What was the CAUSE of this?")
            elif train_df['question'][i] == "effect":
                train_data['text4'].append("What happened as a RESULT?")
            else:
                raise ValueError(f'Invalid question type: {train_df["question"][i]}')
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['premise'].tolist()
        valid_data['text2'] = valid_df['choice1'].tolist()
        valid_data['text3'] = valid_df['choice2'].tolist()
        valid_data['text4'] = []
        for i in range(len(valid_df)):
            if valid_df['question'][i] == "cause":
                valid_data['text4'].append("What was the CAUSE of this?")
            elif valid_df['question'][i] == "effect":
                valid_data['text4'].append("What happened as a RESULT?")
            else:
                raise ValueError(f'Invalid question type: {valid_df["question"][i]}')
        valid_data['label'] = valid_df['label'].tolist()

        text_data['idx'] = test_df['idx'].tolist()
        text_data['text1'] = test_df['premise'].tolist()
        text_data['text2'] = test_df['choice1'].tolist()
        text_data['text3'] = test_df['choice2'].tolist()
        text_data['text4'] = []
        for i in range(len(test_df)):
            if test_df['question'][i] == "cause":
                text_data['text4'].append("What was the CAUSE of this?")
            elif test_df['question'][i] == "effect":
                text_data['text4'].append("What happened as a RESULT?")
            else:
                raise ValueError(f'Invalid question type: {test_df["question"][i]}')
        text_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, text_data, num_classes

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
        # Multiple choice task preprocessing
        for idx in tqdm(range(len(split_data['text1'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            # Get text and label
            premise = split_data['text1'][idx]
            choice1 = split_data['text2'][idx]
            choice2 = split_data['text3'][idx]
            question = split_data['text4'][idx]
            label = split_data['label'][idx]

            # Tokenize
            prompt = f"{premise} {question}"

            tokenized = tokenizer([prompt, prompt], [choice1, choice2],
                                  padding='max_length', truncation=True,
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
