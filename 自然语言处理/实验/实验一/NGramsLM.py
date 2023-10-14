#!/bin/env python

import argparse
from itertools import product
import math
import nltk
from pathlib import Path
from preprocess import preprocess
import random

EOS = "</s>" 
SOS = "<s> "

# 加载数据
def load_data(train_path):
    """Load train corpora
    Newlines will be stripped out. 
    Args:
        train_path (Path) -- the training corpus to use. 
    Returns:
        The train sets, as lists of sentences.
    """   

    with open(train_path, 'r') as f:
        train = [l.strip() for l in f.readlines()]
    
    return train


class LanguageModel(object):
    """An n-gram language model trained on a given corpus.
    
    For a given n and given training corpus, constructs an n-gram language
    model for the corpus by:
    1. preprocessing the corpus (adding SOS/EOS/UNK tokens)
    2. calculating (smoothed) probabilities for each n-gram

    Also contains methods for calculating the perplexity of the model
    against another corpus, and for generating sentences.

    Args:
        train_data (list of str): list of sentences comprising the training corpus.
        n (int): the order of language model to build (i.e. 1 for unigram, 2 for bigram, etc.).
        laplace (int): lambda multiplier to use for laplace smoothing (default 1 for add-1 smoothing).

    """

    def __init__(self, train_data, n, laplace=1):
        self.n = n
        self.laplace = laplace
        self.tokens = preprocess(train_data, n)
        self.vocab  = nltk.FreqDist(self.tokens)
        self.ngram_probabilities = self._smooth()
    def _smooth(self):

        """Apply Laplace smoothing to n-gram frequency distribution.
            
            Here, n_grams refers to the n-grams of the tokens in the training corpus,
            while m_grams refers to the first (n-1) tokens of each n-gram.

            Returns:
                dict: Mapping of each n-gram (tuple of str) to its Laplace-smoothed 
                probability (float).

            """
        n_grams = nltk.ngrams(self.tokens, self.n)
        m_grams = nltk.ngrams(self.tokens, self.n - 1)
        vocab_size = len(self.vocab)
        
        # 计算 n-gram 和 (n-1)-gram 的计数
        n_gram_counts = nltk.FreqDist(n_grams)
        m_gram_counts = nltk.FreqDist(m_grams)
        
        # 应用拉普拉斯平滑来计算概率
        smoothed_probabilities = {}
        for n_gram, count in n_gram_counts.items():
            m_gram = n_gram[:-1]  # 获取 n-gram 的 (n-1)-gram 部分
            numerator = count + self.laplace
            denominator = m_gram_counts[m_gram] + (vocab_size * self.laplace)
            probability = numerator / denominator
            smoothed_probabilities[n_gram] = probability
        return smoothed_probabilities


    def _create_model(self):

        """Create a probability distribution for the vocabulary of the training corpus.
        
        If building a unigram model, the probabilities are simple relative frequencies
        of each token with the entire corpus.

        Otherwise, the probabilities are Laplace-smoothed relative frequencies.

        Returns:
            A dict mapping each n-gram (tuple of str) to its probability (float).

        """
        if self.n == 1:
            # 如果是一元模型，直接无脑计算相对频率
            vocab_size = len(self.vocab)
            probabilities = {token: freq / vocab_size for token, freq in self.vocab.items()}
        else:
            # 如果是高阶 n-gram 模型，用拉普拉斯平滑一下
            probabilities = self._smooth()
        return probabilities

        
    def prob_ngram(self, str):

        """print the probability of the specified n-gram from the pretrained n-gram language model"
        Args:
            str:  the specified n-gram
        """
        # 特殊情况
        if len(str) != self.n:
            raise ValueError(f"Expected an n-gram of length {self.n}, but received an n-gram of length {len(str)}")

        # 从上面得到的ngram_probabilities中获取str的概率
        probability = self.ngram_probabilities.get(str, 0.0)    
        print(f"{str}的概率为: {probability}")

	
    def prob_sent(self, sentence):
        # print the probabiltiy of generating the sentence "sentence" using the pretrained n-gram language modeling
        
        # 标记输入的句子
        tokens = sentence.split()
        # print(tokens)
        # 确保句子至少要有一个N—gram标记
        if len(tokens) < self.n:
            raise ValueError(f"Sentence must contain at least {self.n} tokens for the {self.n}-gram model.")

        # 初始化句子概率
        probability = 1.0
        # 迭代句子并计算 n-gram 概率的乘积
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n]) 
            ngram_prob = self.ngram_probabilities.get(ngram, 0.0) 
            probability *= ngram_prob # 基于马尔可夫过程计算句子概率
        print(f"句子{sentence}'的概率为: {probability}")
	
    # 给出前n-1个标记，预测下一个合适的标记
    def generate_sentences(self, num, min_len=12, max_len=24):
        """Generate num random sentences using the language model.

        Sentences always begin with the SOS token and end with the EOS token.
        While unigram model sentences will only exclude the UNK token, n>1 models
        will also exclude all other words already in the sentence.

        Args:
            num (int): the number of sentences to generate.
            min_len (int): minimum allowed sentence length.
            max_len (int): maximum allowed sentence length.
        Yields:
            A tuple with the generated sentence and the combined probability
            (in log-space) of all of its n-grams.
        """
        for _ in range(num):
            sentence = ['<s>', '<s>']  # 初始化待生成句子
            prob = 0.0  # 初始化概率
            eos_added = False  # 用于标记是否已经添加了EOS
            while True:
                if len(sentence) >= max_len:
                    if not eos_added:
                        sentence.append(EOS)
                    yield (' '.join(sentence), prob)
                    break

                # 选倒数前 n - 1 个单词
                context = tuple(sentence[-(self.n - 1):])
                
                # 获取所有候选的下一个单词
                candidates = self._get_candidate_words(context, without=sentence)
                
                # 随机选择下一个单词
                if candidates:
                    selected_candidate = random.choice(candidates)
                    next_token = selected_candidate[0]
                    next_prob = selected_candidate[1]
                    # 将预测的单词添加到句子中
                    sentence.append(next_token)
                    # 更新句子概率（对数相加）
                    prob += math.log(next_prob)
                    # 句子长度满足要求就可以结束了
                    if (len(sentence) - 3) >= min_len and random.random() < 0.2:
                        if not eos_added:
                            sentence.append(EOS)
                            eos_added = True
                        yield (' '.join(sentence), prob)
                        break
                else:
                    # 如果没有候选词，以概率 1 返回EOS，只添加一次
                    if not eos_added:
                        sentence.append(EOS)
                        eos_added = True
                    yield (' '.join(sentence), prob)
                    break

    def _get_candidate_words(self, context, without=[]):
        # 获取以给定 (n-1) 个标记 "context" 开头的所有 n-gram
        candidates = [ngram for ngram in self.ngram_probabilities if ngram[:-1] == context]
        # 从候选者中删除 "without" 中的标记
        candidates = [ngram for ngram in candidates if ngram[-1] not in without and ngram[-1] not in ["<s>", "</s>", "<UNK>"]]
        # 返回候选的下一个单词和概率
        candidate_words = [(ngram[-1], self.ngram_probabilities[ngram]) for ngram in candidates]
        return candidate_words

if __name__ == '__main__':
    parser = argparse.ArgumentParser("N-gram Language Model")
    parser.add_argument('--data', type=str, default="自然语言处理\实验\实验一\Expertment1_dataset.txt",
            help='Location of the data directory containing ')
    parser.add_argument('--n', type=int, default=3,
            help='Order of N-gram model to create (i.e. 1 for unigram, 2 for bigram, etc.)')
    parser.add_argument('--laplace', type=float, default=0.01,
            help='Lambda parameter for Laplace smoothing (default is 0.01 -- use 1 for add-1 smoothing)')
    
    # args是包含命令行参数值的命名空间对象
    args = parser.parse_args()

    # Load and prepare train/test data
    data_path = Path(args.data)
    train = load_data(data_path)

    print("Loading {}-gram model...".format(args.n))
    lm = LanguageModel(train, args.n, laplace=args.laplace)
    print("Vocabulary size: {}".format(len(lm.vocab)))

    #######################################################################################
    # N-gram概率
    ngram_test = [
        "at a time","at time a","in the end"
    ]
    ngram_test = [tuple(ngram.split()) for ngram in ngram_test]
    [lm.prob_ngram(ngram) for ngram in ngram_test]

    
    # ########################################################################################
    # 生成给定句子的概率
    sentences_test = "the company said it has agreed to sell its shares in a statement"
    lm.prob_sent(sentences_test)

    ########################################################################################
    # 创造句子
    print("Generating sentences...")

    sentences_generator = lm.generate_sentences(num=10, min_len=10, max_len=24)
    for sentence, prob in sentences_generator:
        print(f"句子：{sentence}，概率：{math.exp(prob)}")