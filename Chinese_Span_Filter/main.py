import Chinese_Spam_Filter
import re

# spam类对象
spam = Chinese_Spam_Filter.Filter()

# 保存词频的词典
trash_dict = {}
normal_dict = {}
test_dict = {}

# 保存每封邮件中出现的词
words_list = []
words_dict = {}

# 保存预测结果,key为文件名，值为预测类别
test_result = {}

# 分别获得正常邮件、垃圾邮件及测试文件名称列表
normal_filelist = spam.Get_file_list("./normal")
trash_filelist = spam.Get_file_list("./trash")
test_filelist = spam.Get_file_list("./test")

# 获取训练集中正常邮件与垃圾邮件的数量

normal_filelen = len(normal_filelist)
trash_filelen = len(trash_filelist)

# 获得停用词表，用于对停用词过滤
stop_list = spam.Get_stop_words()

# 获得正常邮件中的词频
print("开始生成正常邮件词频...")
for filename in normal_filelist:
    words_list.clear()

    for email in open("./normal/" + filename, mode='r', encoding='UTF-8'):
        # 过滤掉非中文字符
        # rule=re.compile(r"[^\w\u4e00-\u9fff]+")
        # sub用于替换匹配项
        email = re.sub("[^\w\u4e00-\u9fff]+", "", email)
        email = re.sub("[A-Za-z0-9]", "", email)
        email = re.sub("_", "", email)
        # 将每封邮件出现的词保存在wordsList中
        spam.Get_word_list(stop_list, words_list, email)

    # 统计每个词在所有邮件中出现的次数
    spam.Add_to_dict(words_list, words_dict)

normal_dict = words_dict.copy()
print("成功生成正常邮件词频!")
"""for key, value in list(normal_dict.items())[:10]:
    print(key + ':' + str(value))

a = normal_dict.get("无中生有", 0)
print(a)
b = normal_dict.get("乐不思蜀", 0)
print(b)
"""

# 获得垃圾邮件中的词频
words_dict.clear()
print("开始生成垃圾邮件词频...")
for filename in trash_filelist:
    words_list.clear()

    for email in open("./trash/" + filename, mode='r', encoding='UTF-8'):
        email = re.sub(r"[^\w\u4e00-\u9fff]+", "", email)
        email = re.sub("[A-Za-z0-9]", "", email)
        email = re.sub("_", "", email)
        spam.Get_word_list(stop_list, words_list, email)
    spam.Add_to_dict(words_list, words_dict)

trash_dict = words_dict.copy()
print("成功生成垃圾邮件词频!")
"""
for key,value in trash_dict.items():
    print(key+':'+str(value))
"""

# 测试邮件
print("开始统计测试邮件词频...")
for filename in test_filelist:
    test_dict.clear()
    words_dict.clear()
    words_list.clear()
    for email in open("./test/" + filename, mode='r', encoding='UTF-8'):
        email = re.sub(r"[^\w\u4e00-\u9fff]+", "", email)
        email = re.sub("[A-Za-z0-9]", "", email)
        email = re.sub("_", "", email)
        spam.Get_word_list(stop_list, words_list, email)
    spam.Add_to_dict(words_list, words_dict)
    test_dict = words_dict.copy()

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的100个词
    word_prob_list = spam.Get_test_words(test_dict, trash_dict, normal_dict, normal_filelen, trash_filelen)

    # 对每封邮件得到的100个词计算贝叶斯概率
    p1, p2 = spam.cal_bayes(word_prob_list, trash_dict, normal_dict)
    if (p1 > p2):
        test_result.setdefault(filename, 0)
    else:
        test_result.setdefault(filename, 1)
"""
    if (p1 > 0.9):
        test_result.setdefault(filename, 0)
    else:
        test_result.setdefault(filename, 1)
"""

# 计算分类准确率（测试集中文件名低于1000的为正常邮件）
test_accuracy=spam.cal_accuracy(test_result)
#for i,ic in test_result.items():
#    print(i+"/"+str(ic))
#print(test_accuracy)
print("分类准确率为:%f" % test_accuracy)

