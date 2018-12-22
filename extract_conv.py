import re
import sys
import pickle
from tqdm import tqdm

def make_split(line):
    '''

    判断句子结尾是否有标点符号，有的话返回空，没有就生成一个逗号，
    这个合并两个对话的时候使用
    :param line:
    :return:
    '''

    if re.match(r'.*([，…？！\.,])$',''.join(line)):
        return ''
    return '，'

def good_line(line):
    '''
    这个不多说，容易理解
    :param line:
    :return:
    '''
    if len(re.findall(r'[a-zA-Z0-9]',line))>2:
        return False
    return True

def regular(sentence):
    '''it work！！！！'''
    sentence = re.sub(r'\.{3,100}','…',sentence)
    sentence = re.sub(r'…{2,100}','…',sentence)
    sentence = re.sub(r',{1,100}','，',sentence)
    sentence = re.sub(r'\.{1,100}','。',sentence)
    sentence = re.sub(r'\?{1,100}','？',sentence)
    sentence = re.sub(r'\!{1,100}','！',sentence)
    return sentence

def main(limit=20,x_limit = 3,y_limit = 6):
    from word_sequence import WordSequence

    print('extract lines')
    fp = open("dgk_shooter_min.conv",'r',errors='ignore',encoding='utf-8')
    groups = []
    group = []

    for line in tqdm(fp):
        if line.startswith('M '):
            line = line.replace('\n','')

            if '/' in line:
                line = line[2:].split('/')
            else:
                line = list(line[2:])

            line = line[:-1]
            group.append(regular(''.join(line)))

        else:
            if group:
                groups.append(group)
                group = []

    if group:
        groups.append(group)
        group = []

    print('extracts group')
    x_data = []
    y_data = []
    for group in tqdm(groups):
        for index,data in enumerate(group):#这index就是序号，data就是group中的语句。
            last_line = None
            if index > 0:
                last_line = group[index-1]
                if not good_line(last_line):
                    last_line = None
            #取到上一行

            next_line = None
            if index < len(group) - 1:
                next_line = group[index+1]
                if not good_line(next_line):
                    next_line = None
            #取到下一行

            #取到下下一行
            next_next_line = None
            if index < len(group) - 2:
                next_next_line = group[index+2]

                if not good_line(next_next_line):
                    next_next_line = None

            if next_line:
                x_data.append(group[index])
                y_data.append(next_line)
                #当前行作为问题，下一行作为答案

                if last_line:
                    x_data.append(last_line + make_split(last_line)+group[index])
                    y_data.append(next_line)
                #如果有上一行，那么上一行加当前行作为问，下一行作为答

                if next_next_line:
                    x_data.append(data)
                    y_data.append(next_line + make_split(next_line) + next_next_line)
                #如果有下下一行，当前行做为问，下一行+下下一行作为答

    print(len(x_data),len(y_data))
    for ask,answer in zip(x_data[:20],y_data[:20]):
        print(ask)
        print(answer)
        print('*'*20)

    print('fit data')
    data = list(zip(x_data,y_data))
    data = [(x,y) for x,y in data if len(x)< limit and len(y) < limit and len(x)>=x_limit and len(y) >=y_limit]

    #这里可以看出来需要将对的tupledump到pickle中
    pickle.dump(data,open('chatbot.pkl','wb'))
    ws = WordSequence()
    ws.fit(x_data + y_data)#需要把语句对中的词汇生成一个字典，key是单词，value是字典。
    pickle.dump(ws,open('WordSequence.pkl','wb'))#把字典dump到语句中。
    print('done')







if __name__ == '__main__':
    str1 = ['今','天','天','气','好','a','A','c','？']
    str2 = '今天天气很好哟!?.'
    print(make_split(str1))
    print(good_line(''.join(str1)))
    print(regular(str2))
    main()

