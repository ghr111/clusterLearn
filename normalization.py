"""
@author: liushuchun
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()


def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ''.join(filtered_tokens)
    return filtered_text

# @corpyus是数组，每个元素都是一个汉语句子
def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        #jieba是优秀的中文分词第三方库
        #可以自动把句子分成中文单词，返回值是jieba.lcut List
        #https://www.cnblogs.com/xiaoyh/p/9919590.html
        #字符串拼接可以让list转换为string
        text =" ".join(jieba.lcut(text))
        #list的append把接收的string转换为list中的一个元素
        normalized_corpus.append(text)
    return normalized_corpus