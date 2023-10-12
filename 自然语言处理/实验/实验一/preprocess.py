#!/bin/env python

import nltk

SOS = "<s> "
EOS = "</s>"
UNK = "<UNK>"

def add_sentence_tokens(sentences, n):
    """Wrap each sentence in SOS and EOS tokens.

    For n >= 2, n-1 SOS tokens are added, otherwise only one is added.

    Args:
        sentences (list of str): the sentences to wrap.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        List of sentences with SOS and EOS tokens wrapped around them.

    """
    sos = SOS * (n-1) if n > 1 else SOS
    return ['{}{} {}'.format(sos, s, EOS) for s in sentences]

def replace_singletons(tokens):
    """Replace tokens which appear only once in the corpus with <UNK>.
    
    Args:
        tokens (list of str): the tokens comprising the corpus.
    Returns:
        The same list of tokens with each singleton replaced by <UNK>.
    
    """
    vocab = nltk.FreqDist(tokens)
    return [token if vocab[token] > 1 else UNK for token in tokens]

def preprocess(sentences, n):
    """Add SOS/EOS/UNK tokens to given sentences and tokenize.

    Args:
        sentences (list of str): the sentences to preprocess.
        n (int): order of the n-gram model which will use these sentences.
    Returns:
        The preprocessed sentences, tokenized by words.

    """
    sentences = add_sentence_tokens(sentences, n)#处理后的句子列表
    tokens = ' '.join(sentences).split(' ')#将句子列表变成单词列表
    tokens = replace_singletons(tokens) #将tokens里面出现次数为1的单词替换为"UNK"，其余不变
    return tokens


