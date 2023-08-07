# Welcome!

This is the base code of various Natural Language Processing (NLP) tasks. This project has been inspired by [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial), and I wanted to write a tutorial code for NLP.

## How to start

Prepare a virtual environment (Python 3.8) and install the requirements.

```shell
$ git clone https://github.com/c-juhwan/nlp-basecode
$ cd nlp-basecode
$ conda create -n nlp-basecode python=3.8
$ conda activate nlp-basecode
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
```

After the installation, move to the directory of each task and run the code.

## Content

1. [Text Classification](https://github.com/c-juhwan/nlp-basecode/tree/master/N01_SingleClassification)
2. [Textual Entailment](https://github.com/c-juhwan/nlp-basecode/tree/master/N02_TextualEntailment)
3. [GLUE Benchmark](https://github.com/c-juhwan/nlp-basecode/tree/master/N03_GLUEBenchmark)
4. [SuperGLUE Benchmark](https://github.com/c-juhwan/nlp-basecode/tree/master/N04_SuperGLUE)
5. [Image Classification](https://github.com/c-juhwan/nlp-basecode/tree/master/V01_ImageClassification)
6. More to come!
