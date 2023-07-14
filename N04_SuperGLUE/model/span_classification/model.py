# Standard Library Modules
import os
import sys
import argparse
# Pytorch Modules
import torch
import torch.nn as nn
# Huggingface Modules
from transformers import AutoConfig, AutoModel
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import get_huggingface_model_name

class SpanClassificationModel(nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super(SpanClassificationModel, self).__init__()
        self.args = args

        # Define model
        huggingface_model_name = get_huggingface_model_name(self.args.model_type)
        self.config = AutoConfig.from_pretrained(huggingface_model_name)
        if args.model_ispretrained:
            self.model = AutoModel.from_pretrained(huggingface_model_name)
        else:
            self.model = AutoModel.from_config(self.config)

        self.embed_size = self.model.config.hidden_size
        self.hidden_size = self.model.config.hidden_size
        self.num_classes = self.args.num_classes

        # Define extractor
        # Even though https://github.com/nyu-mll/jiant/blob/daa5a258e3af5e7503288de8401429eaf3f58e13/jiant/proj/main/modeling/heads.py#L120
        # Uses span attention and pooling, we use 1d cnn for simplicity
        self.extractor = nn.Sequential(
            nn.Conv1d(self.embed_size, self.hidden_size, kernel_size=3, padding='same'),
            nn.Dropout(self.args.dropout_rate),
            nn.GELU(),
            nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, padding='same'),
        )

        # Define classifier - custom classifier is more flexible than using BERTforSequenceClassification
        # For example, you can use soft labels for training, etc.
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size), # Cls output + span1 output + span2 output
            nn.Dropout(self.args.dropout_rate),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor, span1_index: list, span2_index: list) -> torch.Tensor:
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
        cls_output = model_output.last_hidden_state[:, 0, :] # (batch_size, hidden_size)

        # Unbatch span indices and extract span outputs
        span1_output, span2_output = [], []
        span1_max_length, span2_max_length = 0, 0
        for idx in range(len(span1_index)): # batch_size

            if len(span1_index[idx]) == 0:
                span1_output.append(torch.zeros(1, self.hidden_size).to(cls_output.device))
            elif len(span1_index[idx]) == 1:
                span1_output.append(model_output.last_hidden_state[0, span1_index[idx][0], :].unsqueeze(0)) # (1, hidden_size)
            else:
                span1_output.append(model_output.last_hidden_state[idx, span1_index[idx][0]:span1_index[idx][-1]+1, :]) # (span1_length, hidden_size)
                if span1_max_length < (span1_index[idx][-1] - span1_index[idx][0] + 1):
                    span1_max_length = span1_index[idx][-1] - span1_index[idx][0] + 1


            if len(span2_index[idx]) == 0:
                span2_output.append(torch.zeros(1, self.hidden_size).to(cls_output.device))
            elif len(span2_index[idx]) == 1:
                span2_output.append(model_output.last_hidden_state[0, span2_index[idx][0], :].unsqueeze(0))
            else:
                span2_output.append(model_output.last_hidden_state[idx, span2_index[idx][0]:span2_index[idx][-1]+1, :]) # (span2_length, hidden_size)
                if span2_max_length < (span2_index[idx][-1] - span2_index[idx][0] + 1):
                    span2_max_length = span2_index[idx][-1] - span2_index[idx][0] + 1

        # Pad span outputs before stacking
        for idx in range(len(span1_output)):
            if span1_output[idx].shape[0] < span1_max_length:
                span1_output[idx] = torch.cat([span1_output[idx], torch.zeros(span1_max_length - span1_output[idx].shape[0], self.hidden_size).to(span1_output[idx].device)], dim=0)
            if span2_output[idx].shape[0] < span2_max_length:
                span2_output[idx] = torch.cat([span2_output[idx], torch.zeros(span2_max_length - span2_output[idx].shape[0], self.hidden_size).to(span2_output[idx].device)], dim=0)

        span1_output = torch.stack(span1_output, dim=0) # (batch_size, span1_length, hidden_size)
        span2_output = torch.stack(span2_output, dim=0) # (batch_size, span2_length, hidden_size)

        span1_output = span1_output.permute(0, 2, 1) # (batch_size, hidden_size, span1_length)
        span2_output = span2_output.permute(0, 2, 1) # (batch_size, hidden_size, span2_length)

        # Extract features
        span1_output = self.extractor(span1_output) # (batch_size, hidden_size, span1_length)
        span2_output = self.extractor(span2_output) # (batch_size, hidden_size, span2_length)

        # Max pooling
        span1_output = torch.max(span1_output, dim=-1)[0] # (batch_size, hidden_size)
        span2_output = torch.max(span2_output, dim=-1)[0] # (batch_size, hidden_size)

        # Concatenate
        final_output = torch.cat([cls_output, span1_output, span2_output], dim=-1) # (batch_size, hidden_size * 3)
        classification_logits = self.classifier(final_output) # (batch_size, num_classes)

        return classification_logits
