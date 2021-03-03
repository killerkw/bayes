import pandas as pd
import jieba
import re
import pickle
import math
import os


class Filter:

    # 加载停用词

    def Get_stop_words(self):
        stoplist = []
        for fp in open("./stop", mode='r', encoding='UTF-8'):
            rs = fp.replace('\n', '')
            stoplist.append(rs)
        return stoplist

    # 生成词典

    def Get_word_list(self, stoplist, wordlist, email):
        temp_list = jieba.lcut(email)
        for fp in temp_list:
            if fp not in stoplist and fp != None and fp.strip() != None:
                if fp not in wordlist:
                    wordlist.append(fp)

    # 统计词频

    def Add_to_dict(self, wordlist, worddict):
        for fp in wordlist:
            if fp not in worddict.keys():
                worddict.setdefault(fp, 1)
            else:
                worddict[fp] += 1

    # 获取文件路径

    def Get_file_list(self, filepath):
        filenames = os.listdir(filepath)
        return filenames

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的100个词
    def Get_test_words(self, test_dict, trash_dict, normal_dict, normal_file_len, trash_file_len):
        word_prob_list = {}
        temp_dict = {}
        for word, num in test_dict.items():
            if word in trash_dict.keys() and word in normal_dict.keys():
                # 该文件中包含词个数
                pw_t = trash_dict[word] / trash_file_len
                pw_n = normal_dict[word] / normal_file_len
                ps_w = pw_t / (pw_t + pw_n)
                word_prob_list.setdefault(word, ps_w)
                # 若某个词只在垃圾邮件出现，就假设它在正常邮件中出现的概率为0.01，反之同理
            if word in trash_dict.keys() and word not in normal_dict.keys():
                pw_t = trash_dict[word] / trash_file_len
                pw_n = 0.01
                ps_w = pw_t / (pw_t + pw_n)
                word_prob_list.setdefault(word, ps_w)
            if word not in trash_dict.keys() and word in normal_dict.keys():
                pw_t = 0.01
                pw_n = normal_dict[word] / normal_file_len
                ps_w = pw_t / (pw_t + pw_n)
                word_prob_list.setdefault(word, ps_w)
            if word not in trash_dict.keys() and word not in normal_dict.keys():
                # 若该词既不在垃圾词典，又不在正常词典，就把条件概率设为0.4
                word_prob_list.setdefault(word, 0.4)
        # 对测试字典里所有词的条件概率进行排序
        sorted(word_prob_list.items(), key=lambda x: x[1], reverse=True)
        for i, (key, value) in enumerate(word_prob_list.items()):
            if i in range(0, 100):
                temp_dict.setdefault(key, value)
        return temp_dict

    # 计算贝叶斯概率
    def cal_bayes(self, word_list, trash_dict, normal_dict):
        pt_w = 1
        pn_w = 1

        for word, prob in word_list.items():
            # print(word + "/" + str(prob))
            pt_w *= (prob)
            pn_w *= (1 - prob)
        p1 = pt_w / (pt_w + pn_w)
        p2 = pn_w / (pt_w + pn_w)
        #         print(str(ps_w)+"////"+str(ps_n))
        return p1, p2

    # 计算预测结果正确率（测试集中文件名低于1000的为正常邮件）
    def cal_accuracy(self, test_result):
        right_count = 0
        error_count = 0
        for name, catagory in test_result.items():
            if (int(name) <= 1000 and catagory == 0) or (int(name) > 1000 and catagory == 1):
                right_count += 1
            else:
                error_count += 1
        return right_count / (right_count + error_count)
