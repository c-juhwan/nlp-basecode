# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data.dataset import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path:str) -> None:
        super(CustomDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.vocab_size = data_['vocab_size']
        self.num_classes = data_['num_classes']
        self.pad_token_id = data_['pad_token_id']

        for idx in tqdm(range(len(data_['input_ids'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'input_ids': data_['input_ids'][idx],
                'attention_mask': data_['attention_mask'][idx],
                'token_type_ids': data_['token_type_ids'][idx],
                'span1_indices': data_['span1_index'][idx],
                'span2_indices': data_['span2_index'][idx],
                'labels': data_['labels'][idx],
                'indices': data_['indices'][idx],
            })

        del data_

    def __getitem__(self, idx:int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(data):
    input_ids = torch.stack([d['input_ids'] for d in data], dim=0) # (batch_size, max_seq_len)
    attention_mask = torch.stack([d['attention_mask'] for d in data], dim=0) # (batch_size, max_seq_len)
    token_type_ids = torch.stack([d['token_type_ids'] for d in data], dim=0) # (batch_size, max_seq_len)
    span1_indices = [d['span1_indices'] for d in data] # list of [span1_index1, span1_index2, ..., span1_indexN]
    span2_indices = [d['span2_indices'] for d in data] # list of [span2_index1, span2_index2, ..., span2_indexN]
    labels = torch.stack([d['labels'] for d in data], dim=0) # (batch_size, num_classes)
    indices = [d['indices'] for d in data] # list of integers

    datas_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'span1_indices': span1_indices,
        'span2_indices': span2_indices,
        'labels': labels,
        'indices': indices,
    }

    return datas_dict
