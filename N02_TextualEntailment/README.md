# Textual Entailment

Textual entailment, also known as Natural Language Inference (NLI) is the task of determining whether a given hypothesis can be inferred from a given premise. For example, given the premise "A person on a horse jumps over a broken down airplane.", the task is to determine whether "Someone is riding a horse." is true, false, or indeterminable. This is a fundamental task in natural language understanding, and more challenging than single text classification.

## Architecture

Pre-trained Language Model (PLM) is a dominant approach for textual entailment task. BERT has [CLS] token at the beginning of the first input sequence, and has [SEP] token between the first and the second input sequence. This structure allows BERT to understand the relationship between two sentences. The [CLS] token is used as the aggregate representation of the first input sequence, and is used for classification of entailment. The [SEP] token is used as the aggregate representation of the second input sequence.

## Implementation

Basically, textual entailment is a text classification task, but not a single text classification. It has two input sequences, and the input is a concatenation of two sequences with [SEP] token between them. The output is a classification result of entailment.

### Dataset

We use SNLI (Stanford Natural Language Inference) dataset. It is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).

## Result

I report the result of models trained on SNLI dataset.

| Model | Accuracy | F1 Score |
| ----- | -------- | -------- |
| BERT  | 0.8910    | 0.8892    |
| ALBERT | 0.8788 | 0.8766 |
| ELECTRA | 0.8977 | 0.8962 |
| DeBERTa | 0.8937 | 0.8925 |
| DeBERTaV3 | 0.9086 | 0.9074 |

## References

## Notes

- Some of the descriptions are written by GPT.
