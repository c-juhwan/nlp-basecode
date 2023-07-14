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
        'span1_word': [],
        'span2_word': [],
        'span1_index_start': [],
        'span1_index_end': [],
        'span2_index_start': [],
        'span2_index_end': [],
        'label': []
    }
    valid_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'span1_word': [],
        'span2_word': [],
        'span1_index_start': [],
        'span1_index_end': [],
        'span2_index_start': [],
        'span2_index_end': [],
        'label': []
    }
    test_data = {
        'idx': [],
        'text1': [],
        'text2': [],
        'span1_word': [],
        'span2_word': [],
        'span1_index_start': [],
        'span1_index_end': [],
        'span2_index_start': [],
        'span2_index_end': [],
        'label': []
    }

    if name in ['wic', 'wic_few']:
        dataset_full = load_dataset('super_glue', 'wic')
        dataset_few = load_dataset('juny116/few_glue', 'wic')

        if name == 'wic':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'wic_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['sentence1'].tolist()
        train_data['text2'] = train_df['sentence2'].tolist()
        train_data['span1_word'] = train_df['word'].tolist()
        train_data['span2_word'] = train_df['word'].tolist()
        train_data['span1_index_start'] = train_df['start1'].tolist()
        train_data['span1_index_end'] = train_df['end1'].tolist()
        train_data['span2_index_start'] = train_df['start2'].tolist()
        train_data['span2_index_end'] = train_df['end2'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['sentence1'].tolist()
        valid_data['text2'] = valid_df['sentence2'].tolist()
        valid_data['span1_word'] = valid_df['word'].tolist()
        valid_data['span2_word'] = valid_df['word'].tolist()
        valid_data['span1_index_start'] = valid_df['start1'].tolist()
        valid_data['span1_index_end'] = valid_df['end1'].tolist()
        valid_data['span2_index_start'] = valid_df['start2'].tolist()
        valid_data['span2_index_end'] = valid_df['end2'].tolist()
        valid_data['label'] = valid_df['label'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['sentence1'].tolist()
        test_data['text2'] = test_df['sentence2'].tolist()
        test_data['span1_word'] = test_df['word'].tolist()
        test_data['span2_word'] = test_df['word'].tolist()
        test_data['span1_index_start'] = test_df['start1'].tolist()
        test_data['span1_index_end'] = test_df['end1'].tolist()
        test_data['span2_index_start'] = test_df['start2'].tolist()
        test_data['span2_index_end'] = test_df['end2'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name in ['wsc', 'wsc_few']:
        dataset_full = load_dataset('super_glue', 'wsc.fixed')
        dataset_few = load_dataset('juny116/few_glue', 'wsc.fixed')

        if name == 'wsc':
            train_df = pd.DataFrame(dataset_full['train'])
        elif name == 'wsc_few':
            train_df = pd.DataFrame(dataset_few['train'])
        valid_df = pd.DataFrame(dataset_full['validation'])
        test_df = pd.DataFrame(dataset_full['test'])
        num_classes = 2

        train_data['idx'] = train_df['idx'].tolist()
        train_data['text1'] = train_df['text'].tolist()
        train_data['text2'] = train_df['text'].tolist()
        train_data['span1_word'] = train_df['span1_text'].tolist()
        train_data['span2_word'] = train_df['span2_text'].tolist()
        train_data['span1_index_start'] = train_df['span1_index'].tolist()
        train_data['span1_index_end'] = train_df['span1_index'].tolist()
        train_data['span2_index_start'] = train_df['span2_index'].tolist()
        train_data['span2_index_end'] = train_df['span2_index'].tolist()
        train_data['label'] = train_df['label'].tolist()

        valid_data['idx'] = valid_df['idx'].tolist()
        valid_data['text1'] = valid_df['text'].tolist()
        valid_data['text2'] = valid_df['text'].tolist()
        valid_data['span1_word'] = valid_df['span1_text'].tolist()
        valid_data['span2_word'] = valid_df['span2_text'].tolist()
        valid_data['span1_index_start'] = valid_df['span1_index'].tolist()
        valid_data['span1_index_end'] = valid_df['span1_index'].tolist()
        valid_data['span2_index_start'] = valid_df['span2_index'].tolist()
        valid_data['span2_index_end'] = valid_df['span2_index'].tolist()
        valid_data['label'] = valid_df['label'].tolist()

        test_data['idx'] = test_df['idx'].tolist()
        test_data['text1'] = test_df['text'].tolist()
        test_data['text2'] = test_df['text'].tolist()
        test_data['span1_word'] = test_df['span1_text'].tolist()
        test_data['span2_word'] = test_df['span2_text'].tolist()
        test_data['span1_index_start'] = test_df['span1_index'].tolist()
        test_data['span1_index_end'] = test_df['span1_index'].tolist()
        test_data['span2_index_start'] = test_df['span2_index'].tolist()
        test_data['span2_index_end'] = test_df['span2_index'].tolist()
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
            'span1_index': [],
            'span2_index': [],
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
            'span1_index': [],
            'span2_index': [],
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
            'span1_index': [],
            'span2_index': [],
            'num_classes': num_classes,
            'vocab_size': config.vocab_size,
            'pad_token_id': tokenizer.pad_token_id
        },
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        for idx in tqdm(range(len(split_data['text1'])), desc=f'Preprocessing {split} data', position=0, leave=True):
            # Get text and label
            text1 = split_data['text1'][idx]
            text2 = split_data['text2'][idx]
            span1_word = split_data['span1_word'][idx]
            span2_word = split_data['span2_word'][idx]
            span1_index_start = split_data['span1_index_start'][idx]
            span1_index_end = split_data['span1_index_end'][idx]
            span2_index_start = split_data['span2_index_start'][idx]
            span2_index_end = split_data['span2_index_end'][idx]
            label = split_data['label'][idx]

            # Tokenize text
            if args.task_dataset in ['wic', 'wic_few']:
                tokenized = tokenizer(text1, text2, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt',
                                      return_offsets_mapping=True)
            elif args.task_dataset in ['wsc', 'wsc_few']:
                tokenized = tokenizer(text1, padding='max_length', truncation=True,
                                      max_length=args.max_seq_len, return_tensors='pt',
                                      return_offsets_mapping=True)

            # Append tokenized data to data_dict
            data_dict[split]['input_ids'].append(tokenized['input_ids'].squeeze())
            data_dict[split]['attention_mask'].append(tokenized['attention_mask'].squeeze())
            if args.model_type in ['bert', 'albert', 'electra', 'deberta', 'debertav3']:
                data_dict[split]['token_type_ids'].append(tokenized['token_type_ids'].squeeze())
            else: # roberta does not use token_type_ids
                data_dict[split]['token_type_ids'].append(torch.zeros(args.max_seq_len, dtype=torch.long))
            data_dict[split]['labels'].append(torch.tensor(label, dtype=torch.long)) # Cross Entropy Loss

            # Get span index of tokenized text
            if args.task_dataset in ['wic', 'wic_few']:
                # For wic task, span index is given as character index
                span1_index_list = []
                span2_index_list = []

                offset_count = 0
                for idx, offset in enumerate(tokenized['offset_mapping'].squeeze()):
                    if offset.tolist() == [0, 0]:
                        offset_count += 1
                        continue # Skip special tokens
                    if offset_count == 1 and span1_index_start <= offset[0] and offset[1] <= span1_index_end: # Ensure that span1 is in text1
                        span1_index_list.append(idx)
                    if offset_count == 2 and span2_index_start <= offset[0] and offset[1] <= span2_index_end: # Ensure that span2 is in text2
                        span2_index_list.append(idx)

            elif args.task_dataset in ['wsc', 'wsc_few']:
                # For wsc task, span index is given as word index
                span1_index_list = []
                span2_index_list = []

                # Tokenize span1_word and span2_word
                span1_word_tokenized = tokenizer(span1_word)
                span2_word_tokenized = tokenizer(span2_word)

                # Find span1_index_start_token using span1_word_tokenized
                span1_token_index_start = tokenized.word_to_tokens(span1_index_start)
                span2_token_index_start = tokenized.word_to_tokens(span2_index_start)

                # Translate (start, end) to list of token index
                span1_index_list = list(range(span1_token_index_start[0], span1_token_index_start[1]))
                span2_index_list = list(range(span2_token_index_start[0], span2_token_index_start[1]))

            data_dict[split]['span1_index'].append(span1_index_list)
            data_dict[split]['span2_index'].append(span2_index_list)

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
