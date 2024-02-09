# Standard Library Modules
import os
import gc
import sys
import pickle
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore FutureWarning for pandas
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
# Huggingface Modules
from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path, get_huggingface_model_name

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    train_data = {
        'src_text': [],
        'tgt_text': []
    }
    valid_data = {
        'src_text': [],
        'tgt_text': []
    }
    test_data = {
        'src_text': [],
        'tgt_text': []
    }

    if name in ['wmt16_de_en', 'wmt16_en_de']:
        dataset = load_dataset('wmt16', 'de-en')

        if name == 'wmt16_de_en':
            source_lang = 'de'
            target_lang = 'en'
        elif name == 'wmt16_en_de':
            source_lang = 'en'
            target_lang = 'de'
    elif name in ['wmt16_cs_en', 'wmt16_en_cs']:
        dataset = load_dataset('wmt16', 'cs-en')

        if name == 'wmt16_cs_en':
            source_lang = 'cs'
            target_lang = 'en'
        elif name == 'wmt16_en_cs':
            source_lang = 'en'
            target_lang = 'cs'
    elif name in ['wmt14_fr_en', 'wmt14_en_fr']:
        dataset = load_dataset('wmt14', 'fr-en')

        if name == 'wmt14_fr_en':
            source_lang = 'fr'
            target_lang = 'en'
        elif name == 'wmt14_en_fr':
            source_lang = 'en'
            target_lang = 'fr'

    for each_translation in tqdm(dataset['train']['translation'], desc='Loading train data'):
        train_data['src_text'].append(each_translation[source_lang])
        train_data['tgt_text'].append(each_translation[target_lang])

    for each_translation in tqdm(dataset['validation']['translation'], desc='Loading valid data'):
        valid_data['src_text'].append(each_translation[source_lang])
        valid_data['tgt_text'].append(each_translation[target_lang])

    for each_translation in tqdm(dataset['test']['translation'], desc='Loading test data'):
        test_data['src_text'].append(each_translation[source_lang])
        test_data['tgt_text'].append(each_translation[target_lang])

    del dataset
    return train_data, valid_data, test_data

def preprocessing(args: argparse.Namespace):
    # Load data
    train_data, valid_data, test_data = load_data(args)

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    # Tokenizer
    if args.model_type in ['lstm', 'gru', 'rnn', 'transformer']:
        # Train word-level tokenizer
        src_tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        src_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        src_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        src_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", 1),
                ("[EOS]", 2),
            ],
        )
        src_trainer = WordLevelTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
        )
        tgt_tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tgt_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        tgt_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(individual_digits=True)])
        tgt_tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[
                ("[BOS]", 1),
                ("[EOS]", 2),
            ],
        )
        tgt_trainer = WordLevelTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"]
        )

        src_tokenizer.train_from_iterator(train_data['src_text'], trainer=src_trainer)
        tgt_tokenizer.train_from_iterator(train_data['tgt_text'], trainer=tgt_trainer)

        src_tokenizer.save(os.path.join(preprocessed_path, 'src_tokenizer.json'))
        tgt_tokenizer.save(os.path.join(preprocessed_path, 'tgt_tokenizer.json'))

        src_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(preprocessed_path, 'src_tokenizer.json'),
                                                bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
        tgt_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(preprocessed_path, 'tgt_tokenizer.json'),
                                                bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]", pad_token="[PAD]")
    elif args.model_type == 't5':
        model_name = get_huggingface_model_name(args.model_type)
        src_tokenizer = AutoTokenizer.from_pretrained(model_name)
        tgt_tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        args.vocab_size = config.vocab_size # Reset vocab_size to the vocab_size of pretrained model
    else:
        raise ValueError(f'Invalid model type: {args.model_type}')

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'src_input_ids': [],
            'src_attention_mask': [],
            'src_token_type_ids': [],
            'tgt_input_ids': [],
            'tgt_attention_mask': [],
            'tgt_token_type_ids': [],
            'src_tokenizer': src_tokenizer,
            'tgt_tokenizer': tgt_tokenizer,
            'vocab_size': args.vocab_size,
            'bos_token_id': src_tokenizer.bos_token_id,
            'eos_token_id': src_tokenizer.eos_token_id,
            'unk_token_id': src_tokenizer.unk_token_id,
            'pad_token_id': src_tokenizer.pad_token_id
        },
        'valid': {
            'src_input_ids': [],
            'src_attention_mask': [],
            'src_token_type_ids': [],
            'tgt_input_ids': [],
            'tgt_attention_mask': [],
            'tgt_token_type_ids': [],
            'src_tokenizer': src_tokenizer,
            'tgt_tokenizer': tgt_tokenizer,
            'vocab_size': args.vocab_size,
            'bos_token_id': src_tokenizer.bos_token_id,
            'eos_token_id': src_tokenizer.eos_token_id,
            'unk_token_id': src_tokenizer.unk_token_id,
            'pad_token_id': src_tokenizer.pad_token_id
        },
        'test': {
            'src_input_ids': [],
            'src_attention_mask': [],
            'src_token_type_ids': [],
            'tgt_input_ids': [],
            'tgt_attention_mask': [],
            'tgt_token_type_ids': [],
            'src_tokenizer': src_tokenizer,
            'tgt_tokenizer': tgt_tokenizer,
            'vocab_size': args.vocab_size,
            'bos_token_id': src_tokenizer.bos_token_id,
            'eos_token_id': src_tokenizer.eos_token_id,
            'unk_token_id': src_tokenizer.unk_token_id,
            'pad_token_id': src_tokenizer.pad_token_id
        }
    }

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        assert len(split_data['src_text']) == len(split_data['tgt_text'])
        for src_text, tgt_text in tqdm(zip(split_data['src_text'], split_data['tgt_text']), desc=f'Preprocessing {split} data'):
            src_tokenized = src_tokenizer(src_text, padding='max_length', truncation=True,
                                            max_length=args.max_seq_len, return_tensors='pt')
            tgt_tokenized = tgt_tokenizer(tgt_text, padding='max_length', truncation=True,
                                            max_length=args.max_seq_len, return_tensors='pt')

            # Append to data_dict
            data_dict[split]['src_input_ids'].append(src_tokenized['input_ids'].squeeze())
            data_dict[split]['src_attention_mask'].append(src_tokenized['attention_mask'].squeeze())
            data_dict[split]['src_token_type_ids'].append(src_tokenized['token_type_ids'].squeeze())
            data_dict[split]['tgt_input_ids'].append(tgt_tokenized['input_ids'].squeeze())
            data_dict[split]['tgt_attention_mask'].append(tgt_tokenized['attention_mask'].squeeze())
            data_dict[split]['tgt_token_type_ids'].append(tgt_tokenized['token_type_ids'].squeeze())

        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
