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
        'passage': [],
        'query': [],
        'entities': [],
        #'entity_spans': [], # Not used
        'answers': [],
    }
    valid_data = {
        'idx': [],
        'passage': [],
        'query': [],
        'entities': [],
        #'entity_spans': [],
        'answers': [],
    }
    test_data = {
        'idx': [],
        'passage': [],
        'query': [],
        'entities': [],
        #'entity_spans': [],
        'answers': [],
    }

    if name in ['record', 'record_few']:
        dataset_full = load_dataset('super_glue', 'record')
        dataset_few = load_dataset('juny116/few_glue', 'record')

        if name == 'record':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'record_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['passage'] = train_df['passage'].tolist()
        train_data['query'] = train_df['query'].tolist()
        train_data['entities'] = train_df['entities'].tolist()
        #train_data['entity_spans'] = train_df['entity_spans'].tolist()
        train_data['answers'] = train_df['answers'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['passage'] = valid_df['passage'].tolist()
        valid_data['query'] = valid_df['query'].tolist()
        valid_data['entities'] = valid_df['entities'].tolist()
        #valid_data['entity_spans'] = valid_df['entity_spans'].tolist()
        valid_data['answers'] = valid_df['answers'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['passage'] = test_df['passage'].tolist()
        test_data['query'] = test_df['query'].tolist()
        test_data['entities'] = test_df['entities'].tolist()
        #test_data['entity_spans'] = test_df['entity_spans'].tolist()
        test_data['answers'] = test_df['answers'].tolist()

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
            'indices': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'entities': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'valid': {
            'indices': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'entities': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
        'test': {
            'indices': [],
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': [],
            'entities': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    # Preprocessing for Record dataset
    # Follow https://github.com/nyu-mll/jiant/blob/daa5a258e3af5e7503288de8401429eaf3f58e13/jiant/tasks/lib/record.py
    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['passage'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            # Get text and label
            index = split_data['idx'][idx]
            passage = split_data['passage'][idx]
            query = split_data['query'][idx]
            entities = split_data['entities'][idx]
            # entity_spans = split_data['entity_spans'][idx]
            answers = split_data['answers'][idx]

            for each_entity in entities:
                filled_query = query.replace('@placeholder', each_entity)

                if each_entity in answers: # 0: False, 1: True
                    label = 1
                else:
                    label = 0

                # Tokenize
                tokenized = tokenizer(passage, query, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt')

                # Append tokenized data to data_dict
                data_dict[split]['indices'].append(index)
                data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
                data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
                if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                    data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
                else: # roberta does not use token_type_ids
                    data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
                data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long))
                data_dict[split]['entities'].append(each_entity)

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
