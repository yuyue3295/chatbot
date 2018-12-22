import random
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def test(params):
    from sequence2sequence import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from word_sequence import WordSequence
    from thread_generator import ThreadedGenerator

    data = pickle.load(open('chatbot.pkl','rb'))
    ws = pickle.load(open('WordSequence.pkl','rb'))
    n_epoch = 40
    batch_size = 128
    x_data = []
    y_data = []
    for i in data:
        x_data.append(i[0])
        y_data.append(i[1])

    print('done')
    print(len(x_data))
    print(len(y_data))
    steps = int(len(x_data)/batch_size) + 1# 取整会把小数点去掉，所以需要加1
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    '''
     tf.ConfigProto()函数用在创建Session的时候，用来对session进行参数配置：
     tf.ConfigProto(allow_soft_placement=True)
     在tf中，通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，
     为了防止这种情况，可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。
     
     1. 记录设备指派情况 :  tf.ConfigProto(log_device_placement=True)
     2. 设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 
     被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的
    '''

    save_path='./model/s2ss_chatbot.ckpt'

    tf.reset_default_graph()
    with tf.Graph().as_default():
        random.seed(0)
        np.random.seed(0)
        tf.set_random_seed(0)

        with tf.Session(config=config) as sess:
            model = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=batch_size,
                **params
            )

            init = tf.global_variables_initializer()
            sess.run(init)
            model.load(sess,'model/s2ss_chatbot.ckpt-4')
            flow = ThreadedGenerator(
                batch_flow([x_data,y_data],ws,batch_size,add_end=[False,True]),
                queue_maxsize=30
            )

            dummy_encoder_inputs = np.array([
                np.array([WordSequence.PAD]) for _ in range(batch_size)
            ])
            dummy_encoder_inputs_length = np.array([1]*batch_size)

            for epoch in range(5,n_epoch+1):
                costs = []
                bar = tqdm(range(steps),total=steps,
                           desc='epoch {}, loss=0.000000'.format(epoch))

                for _ in bar:
                    x,xl,y,yl = next(flow)
                    x = np.flip(x,axis=1)

                    add_loss = model.train(sess,dummy_encoder_inputs,
                                           dummy_encoder_inputs_length,
                                           y,yl,loss_only=True)
                    add_loss *= -0.5

                    cost,lr = model.train(sess,x,xl,y,yl,return_lr=True,add_loss=add_loss)
                    costs.append(cost)

                    bar.set_description('epoch {} loss={:.6f},lr={:.6f}'.format(
                        epoch,
                        np.mean(costs),
                        lr
                    ))
                model.save(sess,save_path='./model/s2ss_chatbot.ckpt',index=epoch)
            flow.close()

            init = tf.global_variables_initializer()

            # 测试
            tf.reset_default_graph()
            model_pred = SequenceToSequence(
                input_vocab_size=len(ws),
                target_vocab_size=len(ws),
                batch_size=1,
                mode='decode',
                beam_width=1,
                parallel_iterations=1,
                **params
            )
            init = tf.global_variables_initializer()
            with tf.Session(config=config) as sess:
                sess.run(init)
                model_pred.load(sess,save_path)

                bar = batch_flow([x_data,y_data],ws,1,add_end=False)
                t= 0
                for x,xl,y,yl in bar:
                    x = np.flip(x,axis=1)
                    pred = model_pred.predict(
                        sess,
                        np.array(x),
                        np.array(xl)
                    )
                    print(ws.inverse_transform(x[0]))
                    print(ws.inverse_transform(y[0]))
                    print(ws.inverse_transform(pred[0]))
                    t += 1
                    if t >= 3:
                        break


def main():
    import json
    test(json.load(open('params.json')))

def test_given_data():
    from sequence2sequence import SequenceToSequence
    from data_utils import batch_flow_bucket as batch_flow
    from word_sequence import WordSequence
    from thread_generator import ThreadedGenerator

    data = pickle.load(open('chatbot.pkl', 'rb'))
    ws = pickle.load(open('WordSequence.pkl', 'rb'))
    n_epoch = 40
    batch_size = 5
    x_data = []
    y_data = []
    for i in data:
        x_data.append(i[0])
        y_data.append(i[1])

    print('done')
    print(len(x_data))
    print(len(y_data))
    steps = int(len(x_data) / batch_size) + 1  # 取整会把小数点去掉，所以需要加1
    flow = ThreadedGenerator(
        batch_flow([x_data, y_data], ws, batch_size, add_end=[False, True]),
        queue_maxsize=30
    )

    for i in range(1):
        datas = next(flow)
        print(datas)

if __name__ == "__main__":
    main()
    # test_given_data()





