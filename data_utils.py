import random
import numpy as np
from tensorflow.python.client import device_lib
from word_sequence import WordSequence

def _get_available_gpus():
    '''
    获取当前GPU的信息
    :return:
    '''

    local_device_protos = device_lib.list_local_devices() #获取当前设备的信息。
    print("打印GPU信息")
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

VOCAB_SIZE_THRESHOLD_CPU = 50000 #临界值

def get_embed_device(vocab_size):
    '''根据输入输出的大小，也即是词汇尺寸的临界值来选择阈值CPU，是在CPU上embedding 还是 GPU上进行embedding'''
    gpus = _get_available_gpus()
    if not gpus or vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    return "/gpu:0"

def transform_sentence(sentence,ws,max_len = None,add_end=False):
    '''
    单独的句子转换
    :param sentence:
    :param ws:
    :param max_len:
    :param add_end:
    :return:
    '''
    encoded = ws.transform(
        sentence,
        max_length = max_len if max_len is not None else len(sentence),
        add_end=add_end
    )#这个encoded就是一个编码好的列表。

    encoded_len = len(sentence) + (1 if add_end else 0)
    if encoded_len > len(encoded):
        encoded_len = len(encoded)

    return encoded,encoded_len

def batch_flow(data,ws,batch_size,raw=False,add_end = True):
    '''
    详细的代码注释和测试用例
    从数据中随机生成batch_size的数据，然后给转化后输出出去
    raw:是否返回原始对象，如果为True，假设结果ret，那么len(ret) == len(data) * 3
        如果为false，那么len(ret) == len(data) * 2
        Q = (q1,q2,q3,...qn)
        A = (a1,a2,a3,...an)
        len(Q) == len(A)
        batch_flow([Q,A],ws,batch_size = 32)
        raw = False:
        next(generator) == q_i_encoded,q_i_len,a_i_encoded,a_i_len
        raw = True:
        next(generator) == q_i_encoded,q_i_len,q_i,a_i_encoded,a_i_len,a_i


    :param data:
    :param ws: ws的数量要和data的数量要保持一致（多个的话） len（data）== len（ws）
    :param batch_size:
    :param raw:这里raw为True的话返回结果中就包含没有编码的原句子，返回为False的话，返回的结果就只有编码后的句子和这个批次中的最大长度。
    :param add_end:
    :return:
    '''

    all_data = list(zip(*data)) # all_data的形式为如[('q1', 'a1'), ('q2', 'a2'), ('q3', 'a3'), ('q4', 'a4'), ('q5', 'a5')]
    if isinstance(ws,(list,tuple)):
        assert len(ws) == len(data),'ws的长度必须等于data的长度 如果 ws 是一个 list 或者 tuple'
    if isinstance(add_end,bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end,(list,tuple))),'add_end不是boolean，就应该是一个list'
        assert len(add_end) == len(data),'如果add_end list(tuple)，那么add_end的长度应该和输入数据的长度相同'

    mul = 2
    if raw:
        mul = 3 #这个mul是用来表示数据的如果mul=2的时候，batches[j*mul] 编码后的序列问或者答，batch[j*mul+1]就表示编码后的序列长度，如果序列为3，则batches[j*mul+2]就是原有的问或者答，通过使用一个连续的2个或者

    while True:
        data_batch = random.sample(all_data,batch_size) # 在all_data数据中随机抽取生成batch_size个数据，具体格式是question answer 元组序列
        batches = [[] for  i in range(len(data) *mul)]  # batches是用来做什么的

        max_lens = [] #这里max_lens 最终有两个最大值长度，一个是answer的，一个是question的
        for j in range(len(data)):# 这里data的长度其实为2，一个是question list 集合，另一个是answer集合
            max_len = max([
                len(x[j]) if hasattr(x[j],'__len__') else 0 #这里是x[j]是否包含__len__这个属性，也就是求长度，一个问答，或者答的长度。这里会遍历所有的问，和所有的答，将最大长度的问的长度和最大长度的答的长度添加叫到max_lens中
                for x in  data_batch
            ]) + (1 if add_end[j] else 0)
            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)): #len(data) 是2 所以j的取值是0和1
                if isinstance(ws,(list,tuple)):
                    w = ws[j]
                else:
                    w = ws

                #添加结束标记（结尾）
                line = d[j]
                if add_end[j] and isinstance(line,(tuple,list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x,x1 = transform_sentence(line,w,max_lens[j],add_end[j]) #当j为零的时候，j*mul 值为零
                    batches[j * mul].append(x)
                    batches[j * mul +1].append(x1)#j*mul值为1
                else:
                    batches[j * mul].append(line)
                    batches[j * mul +1].append(line)
                if raw:
                    batches[j * mul + 2].append(line)
        batches = [np.asarray(x) for  x in batches]
        yield batches


def batch_flow_bucket(data,ws,batch_size,raw=False,add_end = True,
                      n_bucket = 5,bucket_ind=1,debug=False):
    '''
    raw:是否返回原始对象，如果为True，假设结果ret，那么len(ret) == len(data) * 3
        如果为false，那么len(ret) == len(data) * 2
        Q = (q1,q2,q3,...qn)
        A = (a1,a2,a3,...an)
        len(Q) == len(A)
        batch_flow([Q,A],ws,batch_size = 32)
        raw = False:
        next(generator) == q_i_encoded,q_i_len,a_i_encoded,a_i_len
        raw = True:
        next(generator) == q_i_encoded,q_i_len,q_i,a_i_encoded,a_i_len,a_i
    :param data:
    :param ws:
    :param batch_size:
    :param raw:
    :param add_end:
    :param n_bucket:就是把数据分成了多少个bucket
    :param bucket_ind:是指哪一个维度的输入作为bucket的依据
    :param debug:
    :return:
    '''
    all_data = list(zip(*data))
    lenghts = sorted(list(set([len(x[bucket_ind]) for x in all_data]))) #长度列表，

    if n_bucket > len(lenghts):
        n_bucket = len(lenghts)

    splits = np.array(lenghts)[
        (np.linspace(0,1,5,endpoint=False) * len(lenghts)).astype(int)
    ].tolist()
    splits += [np.inf] #np.inf无限大的正整数

    if debug:
        print(splits)

    ind_data = {}
    for x in all_data:
        l = len(x[bucket_ind])
        for ind,s in enumerate(splits[:-1]):
            if l >= s and l<= splits[ind + 1]:
                if ind not in ind_data:
                    ind_data[ind] = []
                ind_data[ind].append(x)
                break

    inds = sorted(ind_data.keys())
    ind_p = [len(ind_data[x]) / len(all_data) for x in inds]

    if debug:
        print(np.sum(ind_p),ind_p)

    if isinstance(ws,(list,tuple)):
        assert len(ws) == len(data),'len(ws)必须等于len(data),ws是list或者tuple'

    if isinstance(add_end,bool):
        add_end = [add_end] * len(data)
    else:
        assert(isinstance(add_end,(list,tuple))),'add_end 不是Boolean，就应该是一个list或者tuple of boolean'
        assert len(add_end) == len(data),'如果add_end是list(tuple)，那么add_end的长度就应该和输入数据的长度是一致的'

    mul = 2
    if raw:
        mul = 3

    while True:
        choice_ind = np.random.choice(inds,p=ind_p)
        if debug:
            print('choice_ind',choice_ind)

        data_batch = random.sample(ind_data[choice_ind],batch_size)
        batches = [[] for i in range(len(data) * mul)]

        max_lens = []
        for j in range(len(data)):
            max_len = max([
                len(x[j]) if hasattr(x[j],'__len__') else  0  for x in data_batch
            ]) + (1 if add_end[j] else 0)

            max_lens.append(max_len)

        for d in data_batch:
            for j in range(len(data)):
                if isinstance(ws,(list,tuple)):
                    w = ws[j]
                else:
                    w = ws
                #添加结尾
                line = d[j]
                if add_end[j] and isinstance(line,(tuple,list)):
                    line = list(line) + [WordSequence.END_TAG]

                if w is not None:
                    x,x1 = transform_sentence(line,w,max_lens[j],add_end[j])
                    batches[j * mul].append(x)
                    batches[j * mul +1].append(x1)
                else:
                    batches[j * mul].append(line)
                    batches[j * mul + 1].append(line)

                if raw:
                    batches[j * mul + 2].append(line)

        batches = [np.asarray(x) for x in batches]
        yield batches






def test_batch_flow():
    from fake_data import generate
    x_data,y_data,ws_input,ws_target = generate(size=10000)
    flow = batch_flow([x_data,y_data],[ws_input,ws_target],4)
    x,x1 ,y,y1 = next(flow)
    print(x.shape,y.shape,x1.shape,y1.shape)


def test_batch_flow_bucket():
    from fake_data import generate
    x_data, y_data, ws_input, ws_target = generate(size=10000)
    flow = batch_flow_bucket([x_data, y_data], [ws_input, ws_target], 4,debug =True)
    for i in range(1):
        x,x1,y,y1 = next(flow)
        print(x.shape, y.shape, x1.shape, y1.shape)





if __name__ == '__main__':
    # print(_get_available_gpus())
    # print(get_embed_device(300000))
    test_batch_flow()
    print(_get_available_gpus())