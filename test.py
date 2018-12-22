import sys
import pickle

import numpy as np
import tensorflow as tf

def test(params):
    from sequence2sequence import SequenceToSequence
    from data_utils import batch_flow

    data = pickle.load(open('chatbot.pkl','rb'))
    ws = pickle.load(open('WordSequence.pkl','rb'))
    x_data = []
    y_data = []
    for i in data:
        x_data.append(i[0])
        y_data.append(i[1])

    print('done')
    print(len(x_data))
    print(len(y_data))

    for x in x_data[:5]:
        print(' '.join(x))

    config = tf.ConfigProto(
        device_count = {'CPU':1,'GPU':0},
        allow_soft_placement = True,
        log_device_placement=False
    )

    save_path = './model/s2ss_chatbot.ckpt-26'
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )

    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess,save_path)
        while True:
            user_text = input('请输入您的句子：')
            if user_text in('exit','quit'):
                exit(0)#中途退出的方式
            x_test = [list(user_text.lower())]
            batchs = batch_flow([x_test],ws,1)
            x,xl = next(batchs)
            print(x,xl)
            # x = np.flip(x,axis=1)
            #
            # print(x,xl)
            pred = model_pred.predict(sess,
                                      np.array(x),
                                      np.array(xl))

            print(ws.inverse_transform(x[0]))
            print('预测的结果如下所示：')

            reply = []
            for p in pred:
                ans =ws.inverse_transform(p)
                reply.extend(ans)
            print(''.join(reply))

def main():
    import json
    test(json.load(open('params.json')))

if __name__ == "__main__":
    main()


