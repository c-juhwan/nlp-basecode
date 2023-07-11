# Text Classification

Text classification is one of the most basic tasks of NLP. It is the process of classifying text strings or documents into different categories, depending upon the contents of the strings. Text classification has a variety of applications, such as detecting user sentiment from a tweet, classifying an email as spam or ham, classifying blog posts into different categories, automatic tagging of customer queries, and so on. This section aims to build a text classification model using PyTorch and Hugging Face Transformers.

## Architecture

Text classification model can be constructed by various architectures. However, the core idea is to summarize the input text into a single vector representation, and then use this vector to classify the text. The vector is often called the context vector.

### Recurrent Neural Network (RNN) Family

The most basic approach to classify a given text is using a recurrent neural network (RNN). RNNs are a class of neural networks that are useful for modeling sequence data such as time series or natural language. RNNs have a hidden state that depends on the previous hidden state and the current input. The hidden state can capture information about the past sequence of inputs. In this way, RNNs can take sequences of varying lengths as input to produce a sequence of outputs.

In the code of this section, I used last hidden state of RNN as the context vector.

### Convolutional Neural Network (CNN)

Even though CNN is mostly used for computer vision tasks, it can also be used for text understanding. I followed the architecture of [Yoon Kim (2014)](https://arxiv.org/abs/1408.5882) to build a CNN-based text classification model.

### Transformer

Transformer is a powerful architecture for various NLP tasks, based on self-attention mechanism. I used Transformer encoder to extract features from the input text, and applied global average pooling upon the output of the encoder and obtain context vector.

### Pre-trained Language Model (PLM) Family

Pre-trained Language Model (PLM) is a powerful, conventional model for various NLP tasks. BERT is one of the most famous auto-encoder PLM. BERT has [CLS] token at the beginning of the input sequence, and the output of the [CLS] token is used as the context vector. I used BERT and its successors, such as RoBERTa, ELECTRA, ALBERT, and DeBERTa to build a text classification model.

## Implementation

### Dataset

I used various datasets for text classification task, and they are mainly from [Hugging Face Datasets](https://huggingface.co/datasets). The datasets are as follows:

- [IMDB](https://huggingface.co/datasets/imdb)
- [SST-2](https://huggingface.co/datasets/SetFit/sst2)
- [SST-5](https://huggingface.co/datasets/SetFit/sst5)
- [CoLA](https://huggingface.co/datasets/linxinyuan/cola)
- [TREC](https://huggingface.co/datasets/trec)
- [SUBJ](https://huggingface.co/datasets/SetFit/subj)
- [CR](https://huggingface.co/datasets/SetFit/SentEval-CR)
- [MR](https://aclanthology.org/P05-1015.pdf)
- [ProsCons](https://aclanthology.org/C08-1031.pdf)
- [AG_News](https://huggingface.co/datasets/ag_news)
- [DBpedia](https://huggingface.co/datasets/dbpedia_14)
- [Yelp_Polarity](https://huggingface.co/datasets/yelp_polarity)
- [Yelp_Full](https://huggingface.co/datasets/yelp_review_full)
- [Yahoo_Answers](https://huggingface.co/datasets/yahoo_answers_topics)

## Result

I report the result of models trained on the SST-2 dataset. The result is as follows:

| Model | Accuracy | F1 Score |
| ----- | -------- | -------- |
| RNN   | 0.7161    | 0.7112    |
| CNN   | 0.7861    | 0.7811    |
| GRU   | 0.7778    | 0.7722   |
| LSTM   | 0.7657    | 0.7590    |
| Transformer | 0.7326 | 0.7272 |
| BERT  | 0.9100    | 0.9071    |
| RoBERTa | 0.9309 | 0.9285 |
| ALBERT | 0.8958 | 0.8930 |
| ELECTRA | 0.9391 | 0.9369 |
| DeBERTa | 0.9232 | 0.9208 |
| DeBERTaV3 | 0.9441 | 0.9420 |

## References

- [Yoon Kim. "Convolutional Neural Networks for Sentence Classification" EMNLP 2014](https://aclanthology.org/D14-1181.pdf)
- [Vaswani et al. "Attention Is All You Need" NIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" NAACL 2019](https://aclanthology.org/N19-1423.pdf)
- [Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach" arXiv 2019](https://arxiv.org/abs/1907.11692)
- [Clark et al. "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" ICLR 2020](https://openreview.net/pdf?id=r1xMH1BtvB)
- [Lan et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" ICLR 2020](https://openreview.net/pdf?id=H1eA7AEtvS)
- [He et al. "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" ICLR 2021](https://openreview.net/pdf?id=XPZIaotutsD)
- [He et al. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing" ICLR 2023](https://openreview.net/pdf?id=sE7-XhLxHA)

## Notes

- Some of the descriptions are written by GPT.
