import os
import jieba
from collections import Counter
from jieba.analyse import ChineseAnalyzer
import json
import numpy as np
import pandas as pd
from whoosh.analysis import Tokenizer, Token
from whoosh.compat import text_type
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh import qparser, scoring

# 第一階段比賽使用資料，使用第二階段請更改路徑
corpus_path = "./NC_2.csv"
test_data_path = "./QS_2.csv"
url2content_path = "./url2content_60W.json"

with open("./stop_word.txt", 'r', encoding='utf-8') as text_file:
    stop_words = text_file.read().split()
    stop_words.sort(key=lambda s: len(s), reverse=True)

with open("./same_word.txt", 'r', encoding='utf-8') as text_file:
    same_words = []
    for line in text_file.readlines():
        same_words.append(line.strip().split())

with open("./neg_word.txt", 'r', encoding='utf-8') as text_file:
    neg_words = text_file.read().split()
    neg_words.sort(key=lambda s: len(s), reverse=True)


def full2half(text):
    n = []
    for char in text:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)


def replace(text):
    for index in range(len(same_words)):
        text = text.replace(same_words[index][1], same_words[index][0])
    text = text.lower().replace("\n", "").replace("\t", "").replace("\r", "")
    return full2half(text)


def read_corpus_and_process(file_path=corpus_path):
    # 透過url2content檔，將url映射到文章內容
    # doc_list: 包含語料庫中所有文章的文字內容
    corpus_df = pd.read_csv(file_path)
    url_to_contents = json.load(open(url2content_path))
    doc_indexs = corpus_df["News_Index"]
    doc_urls = corpus_df["News_URL"]
    doc_list = [url_to_contents[url] for url in doc_urls]

    writer = ix.writer()

    # 遍歷語料庫中所有文章，一一進行前處理
    for doc_idx, doc in enumerate(doc_list):
        writer.add_document(path=doc_indexs[doc_idx], content=replace(doc))
        print(doc_idx)

    writer.commit()


def read_query_and_process(file_path=test_data_path):
    query_df = pd.read_csv(file_path)
    query_list = query_df["Query"]

    # 遍歷所有query，一一進行前處理
    for i in range(len(query_list)):
        query_list[i] = replace(query_list[i])

    query_idx = query_df["Query_Index"]

    return query_idx, query_list


def query_expansion(n = 300, t = 10):
    for i in range(len(query_list)):
        concat_all = []
        for content in content_list[i][:n]:
            concat_all += jieba.lcut_for_search(content)

        word_counter = Counter(concat_all)
        word_counter = word_counter.most_common()
        com_words = []
        count = 0
        for ele in word_counter:
            if ele[0] not in stop_words and ele[0] not in query_list[i] and ele[0] != " ":
                for word in com_words[:]:
                    if word in ele[0]:
                        com_words.remove(word)
                        count -= 1
                com_words.append(ele[0])
                count += 1
            if count == t:  
                print(query_list[i])
                print(word_counter[:50])
                print(com_words)
                query_list[i] = query_list[i] + ' ' + ' '.join(com_words)
                break


def dump(rank_list, pred_file_name="./oooo.csv"):
    header = ["Rank_"+str(i).zfill(3)
              for i in range(1, 300+1)]  # 建立輸出檔的header
    df = pd.DataFrame(rank_list, columns=header)  # 將結果存在dataframe中
    df.insert(loc=0, column="Query_Index",
              value=query_idx)  # 插入Query_Index欄位
    df.to_csv(pred_file_name, index=False)  # 將dataframe輸出成csv檔


jieba.initialize()
jieba.del_word('旺旺中')
jieba.del_word('第三天')
jieba.del_word('髮')
jieba.del_word('退之')
jieba.add_word('早收清單', tag='N')
jieba.add_word('天然氣', tag='N')
jieba.add_word('旺旺', tag='N')
jieba.add_word('中時', tag='N')
jieba.add_word('髮', tag='N')
jieba.add_word('月退', tag='N')
jieba.add_word('十八趴', tag='N')

schema = Schema(path=ID(stored=True), content=TEXT(
    stored=True, analyzer=ChineseAnalyzer()))

print("start loading content...")
if not os.path.exists("index2"):
    os.mkdir("index2")
    ix = create_in("index2", schema)
    read_corpus_and_process()
else:
    ix = open_dir("index2")

print('start loading query')

query_idx, query_list = read_query_and_process()

print("start 1 searching ...")
parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
nparser = QueryParser("content", ix.schema, group=qparser.NotGroup)
rank_list = [[] for i in range(len(query_list))]
content_list = [[] for i in range(len(query_list))]
with ix.searcher(weighting=scoring.BM25F(B=0.9, K1=1.2)) as searcher:
    for i in range(len(query_list)):
        query = parser.parse(query_list[i])
        neg_query = parser.parse(' '.join(neg_words))
        if any(word in query_list[i] for word in neg_words):
            query = query | neg_query
        else:
            for word in neg_words:
                query = query | nparser.parse(word)
        print(query)
        results = searcher.search(query, limit=300)
        for hit in results:
            rank_list[i].append(hit['path'])
            content_list[i].append(hit['content'])
        print(len(rank_list[i]))

dump(rank_list, pred_file_name="./ans.csv")

print('query expansion 1')

query_expansion(300,5)

print("start 2 searching ...")
parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
rank_list = [[] for i in range(len(query_list))]
with ix.searcher(weighting=scoring.BM25F(B=0.9, K1=1.2)) as searcher:
    for i in range(len(query_list)):
        query = parser.parse(query_list[i])
        neg_query = parser.parse(' '.join(neg_words))
        if any(word in query_list[i] for word in neg_words):
            query = query | neg_query
        else:
            for word in neg_words:
                query = query | nparser.parse(word)
        print(query)
        results = searcher.search(query, limit=300)
        for hit in results:
            rank_list[i].append(hit['path'])
        print(len(rank_list[i]))

dump(rank_list, pred_file_name="./ans2.csv")

print('query expansion 2')

query_expansion(100,3)

print("start 3 searching ...")
parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
rank_list = [[] for i in range(len(query_list))]
with ix.searcher(weighting=scoring.BM25F(B=0.9, K1=1.2)) as searcher:
    for i in range(len(query_list)):
        query = parser.parse(query_list[i])
        neg_query = parser.parse(' '.join(neg_words))
        if any(word in query_list[i] for word in neg_words):
            query = query | neg_query
        else:
            for word in neg_words:
                query = query | nparser.parse(word)
        print(query)
        results = searcher.search(query, limit=300)
        for hit in results:
            rank_list[i].append(hit['path'])
        print(len(rank_list[i]))

dump(rank_list, pred_file_name="./ans3.csv")

query_expansion(50,1)

print("start 4 searching ...")
parser = QueryParser("content", ix.schema, group=qparser.OrGroup)
rank_list = [[] for i in range(len(query_list))]
with ix.searcher(weighting=scoring.BM25F(B=0.9, K1=1.2)) as searcher:
    for i in range(len(query_list)):
        query = parser.parse(query_list[i])
        neg_query = parser.parse(' '.join(neg_words))
        if any(word in query_list[i] for word in neg_words):
            query = query | neg_query
        else:
            for word in neg_words:
                query = query | nparser.parse(word)
        print(query)
        results = searcher.search(query, limit=300)
        for hit in results:
            rank_list[i].append(hit['path'])
        print(len(rank_list[i]))

dump(rank_list, pred_file_name="./ans4.csv")
