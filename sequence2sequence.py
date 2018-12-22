import numpy as np
import tensorflow as tf
from tensorflow import layers

from tensorflow.python.ops import array_ops
'''
官网上说tensorflow.python.ops是在tensorflow空间对原始python功能（包括numpy等）的wrapper

'''
from tensorflow.contrib import seq2seq
'''
创建一个seq2seq模型，以及动态解码

'''
from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.seq2seq import LuongAttention
from tensorflow.contrib.seq2seq import AttentionWrapper
from tensorflow.contrib.seq2seq import BeamSearchDecoder

from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn.rnn_cell import GRUCell
from tensorflow.nn.rnn_cell import MultiRNNCell
from tensorflow.nn.rnn_cell import DropoutWrapper
from tensorflow.nn.rnn_cell import ResidualWrapper

from word_sequence import WordSequence
from data_utils import get_embed_device

class SequenceToSequence(object):
    '''
    基本流程

    __init__ 基本参数的保证，参数验证(验证参数的合法性)
    build_model 构建模型
    inti_placeholders 初始化一些Tensorflow的变量占位符

    build_singel_cell
    build_decoder_cell

    init_optimizer 如果实在训练模式下运行，那么则需要初始化优化器
    train 训练一个batch数据
    predict 预测一个batch数据
    '''

    def __init__(self,
                 input_vocab_size,# 输入词表的大小
                 target_vocab_size,#输出词表的大小
                 batch_size=32, #数据batch的大小
                 embedding_size=300, #输入词表和输出词表embedding的大小维度，这个是word2vector 这个就是嵌入向量的维度
                 mode='train',#取值为train，代表训练模式，取值为inference，代表的是预测模式!!!!
                 hidden_units=256, #这个rnn或者lstm中隐藏层的大小，encoder和decoder是相同的
                 depth=1, #encoder和decoder rnn的层数，
                 beam_width=0, #是beamsearch的超参数，用于解码
                 cell_type='lstm', #rnn的神经元类型，lstm，gru
                 dropout=0.2, #神经元随机失活的概率
                 use_dropout=False,#是否使用dropout
                 use_residual=False,#是否使用residual
                 optimizer='adam',# 使用哪个优化器
                 learning_rate=1e-3,#学习率
                 min_learing_rate=1e-6,#最小的学习率
                 decay_step=50000,#衰减的步数
                 max_gradient_norm=5.0, #梯度正则裁剪的系数
                 max_decode_step=None, #最大的decode长度，可以非常大
                 attention_type='Bahdanau',#使用的attention类型
                 bidirectional = False,#是否是双向的encoder
                 time_major=False,#是否在计算过程中使用时间作为主要的批量数据！！！
                 seed=0,#一些层间操作的随机数
                 parallel_iterations=None,# 并行执行rnn循环的次数
                 share_embedding=False,#是否让encoder和decoder共用一个embedding
                 pretrained_embedding=False): #是否需要使用预训练的embedding
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units
        self.depth = depth
        self.cell_type = cell_type.lower()
        self.keep_prob = 1.0 - dropout
        self.use_dropout = use_dropout
        self.use_residual = use_residual
        self.attention_type = attention_type
        self.mode = mode
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learing_rate
        self.decay_step = decay_step
        self.max_gradient_norm = max_gradient_norm
        self.bidirectional = bidirectional
        self.seed = seed
        self.pretrain_embedding = pretrained_embedding


        if isinstance(parallel_iterations,int):
            self.parallel_iterations = parallel_iterations
        else:
            self.parallel_iterations = batch_size

        self.time_major = time_major
        self.share_embedding = share_embedding

        #生成均匀分布的随机数，有四个参数，最小值、最大值、随机的种子数（可以为空），类型
        self.initializer = tf.random_uniform_initializer(
            -0.05,0.05,dtype=tf.float32
        )

        assert self.cell_type in ('gru','lstm'),'cell_type 应该是GRU 或者 LSTM'

        if share_embedding:
            assert input_vocab_size == target_vocab_size,'如果share_embedding为True，那么两个vocab_size必须相等'

        assert mode in ('train','decode'),'mode 必须是train或者decode,而不是{}'.format(mode)
        assert dropout >=0 and dropout < 1,'dropout 的值必须大于等于零小于1'
        assert attention_type.lower() in ('bahdanau','luong'),'attention_type必须是bahdanau或者luong，而不是{}'.format(attention_type.lower())
        assert beam_width < target_vocab_size,'beam_width {}应该小于target_vocab_size {}'.format(beam_width,target_vocab_size)# 这个需要验证参数对不对

        self.keep_prob_placeholder = tf.placeholder(
            tf.float32,
            shape=[],
            name = 'keep_prob'
        )

        self.global_step = tf.Variable(
            0,trainable=False,name='global_step'
        )

        self.use_beamsearch_decode = False
        self.beam_width = beam_width
        self.use_beamsearch_decode = True if self.beam_width >0 else False
        self.max_decode_step = max_decode_step

        assert self.optimizer.lower() in ('adadelta','adam','rmsprop','momentum','sgd'),\
        'optimizer必须是下列之一：adadelta,adam,rmsprop,momentum,sgd'

        self.build_model()

    def build_model(self):
        '''
        1、初始化训练、预测所需要的变量
        2、构建编码器（encoder）
        3、构建解码器（decoder）
        4、构建优化器（optimizer）
        5、保存
        :return:
        '''
        self.init_placeholders()
        encoder_outputs,encoder_state = self.build_encoder()
        self.build_decoder(encoder_outputs,encoder_state)

        #为了代码的低耦合，和分离的原则。每一步都写成一个函数。
        if self.mode == 'train':
            self.init_optimizer()

        self.saver = tf.train.Saver()



    def init_placeholders(self):
        '''
        有哪些需要进行输入的，add_loss,编码器的输入，编码器输入长度的输入，解码器的输入，
        解码器的reward，解码器长度的输入，实际训练时解码器的输入，start_token + 解码器的输入

        :return:
        '''
        self.add_loss = tf.placeholder(
            dtype=tf.float32,
            name = 'add_loss'
        )

        #编码器的输入
        self.encoder_inputs = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,None),
            name = 'encoder_inputs'
        )

        #编码器的长度输入
        self.encoder_inputs_length = tf.placeholder(
            dtype=tf.int32,
            shape=(self.batch_size,),
            name='encoder_input_length'
        )

        if self.mode == 'train':

            #解码器的输入
            self.decoder_inputs = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,None),
                name = 'decoder_inputs'
            )


            #解码器输入的rewards，这个rewards会用于强化学习，或者训练的时候使用。
            self.rewards = tf.placeholder(
                dtype=tf.float32,
                shape=(self.batch_size,1),
                name='rewards'
            )

            #解码器长度的输入
            self.decoder_inputs_length = tf.placeholder(
                dtype=tf.int32,
                shape=(self.batch_size,),
                name='decoder_inputs_length'
            )

            self.decoder_start_token = tf.ones(
                shape=(self.batch_size,1),
                dtype=tf.int32
            ) * WordSequence.START #这里decoder_start_token乘以WordSequence.START 里面的ones，变成2

            #实际训练时解码器的输入，start_token + decoder_input
            self.decoder_inputs_train = tf.concat(
                [self.decoder_start_token,
                self.decoder_inputs],
                axis = 1
            ) #解码器的输入需要将start_token 和decoder_input拼接起来。

    def build_single_cell(self,n_hidden,use_residual):
        '''
        构建一个单独的rnn cell
        :param n_hidden: 隐藏层的神经单元数量
        :param use_residual: 是否使用residual wrapper
        :return:
        '''

        if self.cell_type == 'gru':
            cell_type = GRUCell
        else:
            cell_type = LSTMCell

        cell = cell_type(n_hidden)

        #使用self.use_dropout 可以避免过拟合，等等。
        if self.use_dropout:
            cell = DropoutWrapper(
                cell,
                dtype=tf.float32,
                output_keep_prob=self.keep_prob_placeholder,
                seed = self.seed #一些层之间操作的随机数
                )
        #使用ResidualWrapper进行封装可以避免一些梯度消失或者梯度爆炸
        if use_residual:
            cell = ResidualWrapper(cell)
        return cell


    def build_encoder_cell(self):
        '''
        构建单独的编码器cell。
        根据深度，需要多少层网络。
        :return:
        '''

        multi_cell =  MultiRNNCell([
            self.build_single_cell(
                self.hidden_units,
                use_residual=self.use_residual
            )
            for _ in range(self.depth)
        ]
        )

        """RNN cell composed sequentially of multiple simple cells.

        Example:

        ```python
        num_units = [128, 64]
        cells = [BasicLSTMCell(num_units=n) for n in num_units]
        stacked_rnn_cell = MultiRNNCell(cells)
        ```
        """
        # num_units = []
        # for i in range(self.depth):
        #     num_units.append(self.hidden_units)
        # print('num_units 的数目',num_units)
        #
        # cells = [self.build_single_cell(n_hidden=n,use_residual=self.use_residual) for n in num_units]
        # print(cells,'shifou为None')
        # print(tuple(cell.state_size for cell in cells))
        # print(cells[-1].output_size)
        #
        # multi_cell = MultiRNNCell(cells)


        print("in build_encoder_cell")
        print(hasattr(multi_cell,'output_size'))
        print(hasattr(multi_cell,'state_size'))
        return multi_cell


    def build_encoder(self):
        '''
        构建编码器
        编码器的cell，初始化embedding_matrix
        :return:
        '''
        with tf.variable_scope('encoder'):
            encoder_cell = self.build_encoder_cell() #首先是创建编码器，根据层数，使用MultiRNNCell

            with tf.device(get_embed_device(self.input_vocab_size)):
                if self.pretrain_embedding:
                    self.encoder_embeddings = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=(self.input_vocab_size,self.embedding_size)
                        ),
                        trainable = True,
                        name = 'embeddings'
                    )

                    self.encoder_embeddings_placeholder = tf.placeholder(
                        tf.float32,
                        (self.input_vocab_size,self.embedding_size)
                    )

                    self.encoder_embeddings_init = self.encoder_embeddings.assign(
                        self.encoder_embeddings_placeholder
                    )
                else:
                    self.encoder_embeddings = tf.get_variable(
                        name = 'embeddings',
                        shape=(self.input_vocab_size,self.embedding_size),
                        initializer=self.initializer,
                        dtype=tf.float32
                    )

                    '''
                   上面是初始化embedding-matrix 是否加载预先训练好的embedding matrix来进行操作。
                    '''

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                params=self.encoder_embeddings,
                ids=self.encoder_inputs
            )

            if self.use_residual:
                self.encoder_inputs_embedded = layers.dense(self.encoder_inputs_embedded,
                                                            self.hidden_units,
                                                            use_bias=False,
                                                            name='encoder_residual_projection')

            inputs = self.encoder_inputs_embedded
            if self.time_major:
                inputs = tf.transpose(inputs,(1,0,2))

            if not self.bidirectional:
                (
                    encoder_outputs,
                    encode_states
                ) = tf.nn.dynamic_rnn(
                    cell = encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True#建议设置为true
                )
                '''
                一般来讲dynamic_rnn它的优点就是，动态rnn的内存可以交换
                '''

            else:
                encoder_cell_bw = self.build_encoder_cell()
                (
                    (encoder_fw_outputs,encoder_bw_outputs),
                    (encoder_fw_state,encoder_bw_state)
                ) =tf.nn.bidirectional_dynamic_rnn(
                    cell_bw = encoder_cell_bw,
                    cell_fw = encoder_cell,
                    inputs=inputs,
                    sequence_length=self.encoder_inputs_length,
                    dtype=tf.float32,
                    time_major=self.time_major,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=True
                )


                #需要把双向的RNN的输出结果进行合并
                encoder_outputs = tf.concat(
                    (encoder_fw_outputs,encoder_bw_outputs),2
                )

                encoder_state = []
                for i in range(self.depth):
                    encoder_state.append(encoder_fw_state[i])
                    encoder_state.append(encoder_bw_state[i])
                encoder_state = tuple(encoder_state)

        return encoder_outputs,encoder_state

    def build_decoder_cell(self,encoder_outputs,encoder_state):
        '''

        构建解码器的cell
        :param encoder_outputs:
        :param encoder_state:
        :return:
        '''
        encoder_input_length = self.encoder_inputs_length
        batch_size = self.batch_size

        if self.bidirectional:
            encoder_state = encoder_state[-self.depth:]

        if self.time_major:
            encoder_outputs = tf.transpose(encoder_outputs,(1,0,2))

        if self.use_beamsearch_decode:
            '''这个tile_batch 会将tensor复制self.beam_with 份，相当于是
            batch的数据变成了原来的self.beam_width 倍
            '''
            encoder_outputs = seq2seq.tile_batch(
                encoder_outputs,multiplier=self.beam_width
            )
            encoder_state = seq2seq.tile_batch(
                encoder_state,multiplier=self.beam_width
            )


            encoder_input_length = seq2seq.tile_batch(
                self.encoder_inputs_length,multiplier=self.beam_width
            )

            #如果使用了beamsearch，那么输入应该是beam_width的倍数乘以batch_size
            batch_size *=self.beam_width


        if self.attention_type.lower() == 'luong':
            self.attention_mechanism = LuongAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_input_length
            )
        else:
            self.attention_mechanism = BahdanauAttention(
                num_units=self.hidden_units,
                memory=encoder_outputs,
                memory_sequence_length=encoder_input_length
            )#这里的memory 觉得传递得有问题，为什么不是encoder_state呢？

        cell = MultiRNNCell(
            [
                self.build_single_cell(
                    self.hidden_units,
                    use_residual=self.use_residual
                )

                for _ in range(self.depth)
            ])

        alignment_history = (
            self.mode != 'train' and not self.use_beamsearch_decode
        )

        def cell_input_fn(inputs,attention):
            '''
            根据attn_input_feeding属性来判断是否在attention计算前进行一次投影的计算
            :param inputs:
            :param attention:
            :return:
            '''

            if not self.use_residual:
                return array_ops.concat([inputs,attention],-1)

            attn_projection = layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')

            '''
            这个attn_projection(array_ops.concat([inputs,attention],-1))我的理解就是
            layers.Dense(self.hidden_units,
                                           dtype=tf.float32,
                                           use_bias=False,
                                           name='attention_cell_input_fn')(array_ops.concat([inputs,attention],-1))
            因为Dense内部实际上是定义了__call__(self): 的方法，因此可以这样使用
            '''
            return attn_projection(array_ops.concat([inputs,attention],-1))


        cell = AttentionWrapper(
            cell=cell,
            attention_mechanism=self.attention_mechanism,
            attention_layer_size=self.hidden_units,
            alignment_history=alignment_history,#这个是attention的历史信息
            cell_input_fn=cell_input_fn,#将attention拼接起来和input拼接起来
            name='Attention_Wrapper'
        )#AttentionWrapper 注意力机制的包裹器

        decoder_initial_state = cell.zero_state(
            batch_size,tf.float32
        )#这里初始化decoder_inital_state

        #传递encoder的状态
        decoder_initial_state = decoder_initial_state.clone(
            cell_state = encoder_state
        )

        return cell,decoder_initial_state


    def build_decoder(self,encoder_output,encoder_state):
       '''
       构建解码器
       :param encoder_output:
       :param encoder_state:
       :return:
       '''

       with tf.variable_scope('decoder') as decoder_scope:#这里是为了调试方便，将参数折叠成一个层。
           (
               self.decoder_cell,
               self.decoder_initial_state
           ) = self.build_decoder_cell(encoder_output,encoder_state)
           #解码器的embedding matrix
           with tf.device(get_embed_device(self.target_vocab_size)):
               if self.share_embedding:
                   self.decoder_embeddings = self.encoder_embeddings

               elif self.pretrain_embedding:
                   self.decoder_embeddings = tf.Variable(
                       tf.constant(
                           0.0,
                           shape=(self.target_vocab_size,
                                  self.embedding_size)
                       ),
                       trainable=True,
                       name = 'embeddings'
                   )

                   self.decoder_embeddings_placeholder = tf.placeholder(
                       dtype=tf.float32,
                       shape=(self.target_vocab_size,self.embedding_size),
                   )

                   self.decoder_embeddings_init = self.decoder_embeddings.assign(self.decoder_embeddings_placeholder)
               else:
                   self.decoder_embeddings = tf.get_variable(
                       name='embeddings',
                       shape = (self.target_vocab_size,self.embedding_size),
                       initializer=self.initializer,
                       dtype=tf.float32
                   )

                   #上面也是对用于解码器的embedding的初始化
           #定义输出的projection，实际上就是全连接层
           self.decoder_output_projection = layers.Dense(
               self.target_vocab_size,
               dtype=tf.float32,
               use_bias=False,
               name='decoder_output_projection'
           )

           if self.mode == 'train':
               self.decoder_inputs_embedded = tf.nn.embedding_lookup(
                   params=self.decoder_embeddings,
                   ids=self.decoder_inputs_train
               )

               inputs = self.decoder_inputs_embedded
               if self.time_major:
                   inputs = tf.transpose(inputs,(1,0,2))

               #seq2seq的一个类，用来帮助feeding参数。
               training_helper = seq2seq.TrainingHelper(
                   inputs=inputs,
                   sequence_length=self.decoder_inputs_length,
                   time_major=self.time_major,
                   name='training_helper'
               )

               training_decoder = seq2seq.BasicDecoder(
                   cell=self.decoder_cell,
                   helper=training_helper,
                   initial_state=self.decoder_initial_state
               )

              #decoder在当前的batch下的最大time_steps
               max_decoder_length =tf.reduce_max(
                   self.decoder_inputs_length
               )

               (
                   outputs,
                   self.final_state,
                   final_sequence_lengths
                ) = seq2seq.dynamic_decode(
                   decoder=training_decoder,
                   output_time_major=self.time_major,
                   impute_finished=True,
                   maximum_iterations=max_decoder_length,
                   parallel_iterations=self.parallel_iterations,
                   swap_memory=True,
                   scope = decoder_scope
               )

               self.decoder_logits_train = self.decoder_output_projection(
                   outputs.rnn_output
               )

               '''
               self.masks感觉有用，通过这个mask来区分数据位和填充位，这个是计算sequence_loss需要传入的参数。
               
               
               
               '''
               self.masks = tf.sequence_mask(
                   lengths=self.decoder_inputs_length,
                   maxlen=max_decoder_length,
                   dtype=tf.float32,
                   name='masks'
               )

               decoder_logits_train = self.decoder_logits_train
               if self.time_major:
                   decoder_logits_train = tf.transpose(decoder_logits_train(1,0,2))

               self.decoder_pre_train =tf.argmax(
                    decoder_logits_train,
                    axis=-1,
                    name='deocder_pred_train'
                )

               self.tran_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                   labels=self.decoder_inputs,
                   logits=decoder_logits_train
               )

               self.masks_rewards = self.masks * self.rewards

               self.loss_rewards = seq2seq.sequence_loss(
                   logits=decoder_logits_train,
                   targets=self.decoder_inputs,
                   weights=self.masks_rewards,
                   average_across_timesteps=True,
                   average_across_batch=True
               )

               self.loss = seq2seq.sequence_loss(
                   logits=decoder_logits_train,
                   targets=self.decoder_inputs,
                   weights=self.masks,
                   average_across_timesteps=True,
                   average_across_batch=True
               )

               print('in build_decoder')
               print(self.add_loss.name)
               self.add_loss =  self.add_loss + self.loss
               print(self.add_loss.name)

           elif self.mode == 'decode':
               start_tokens = tf.tile(
                   [WordSequence.START],
                   [self.batch_size]
               )
               end_token = WordSequence.END

               def embed_and_input_proj(inputs):
                   return tf.nn.embedding_lookup(
                       self.decoder_embeddings,
                       inputs
                   )

               if not self.use_beamsearch_decode:
                   decoding_helper = seq2seq.GreedyEmbeddingHelper(
                       start_tokens=start_tokens,
                       end_token=end_token,
                       embedding=embed_and_input_proj
                   )

                   inference_decoder = seq2seq.BasicDecoder(
                       cell=self.decoder_cell,
                       helper=decoding_helper,
                       initial_state=self.decoder_initial_state,
                       output_layer=self.decoder_output_projection
                   )

               else:
                   #这里的BeamSearchDecoder 传入的initial_state是经过变换，成了原来的beam_width 这么多倍。
                   inference_decoder = BeamSearchDecoder(
                       cell=self.decoder_cell,
                       embedding=embed_and_input_proj,
                       start_tokens=start_tokens,
                       end_token = end_token,
                       initial_state=self.decoder_initial_state,
                       beam_width=self.beam_width,
                       output_layer=self.decoder_output_projection
                   )
               if self.max_decode_step is not None:
                   max_decoder_step = self.max_decode_step
               else:
                   max_decoder_step = tf.round(
                       tf.reduce_max(self.encoder_inputs_length)* 4
                   )

               (
                   self.decoder_outputs_decode,
                   self.final_state,
                   final_sequence_lengths
               ) = (seq2seq.dynamic_decode(
                   decoder=inference_decoder,
                   output_time_major=self.time_major,
                   maximum_iterations=self.parallel_iterations,
                   swap_memory=True,
                   scope = decoder_scope
               ))

               if not self.use_beamsearch_decode:
                   dod = self.decoder_outputs_decode
                   self.decoder_pred_decode = dod.sample_id
                   self.decoder_pred_decode = tf.transpose(
                       self.decoder_pred_decode,(1,0)
                   )
               else:
                   self.decoder_pred_decode = self.decoder_outputs_decode.predicted_ids
                   if self.time_major:
                       self.decoder_pred_decode = tf.transpose(
                           self.decoder_pred_decode,(1,0,2)
                       )

                   self.decoder_pred_decode = tf.transpose(
                       self.decoder_pred_decode,(1,0,2)
                   )

                   self.decoder_pred_decode = tf.transpose(
                       self.decoder_pred_decode,
                       perm=[0,2,1]
                   )

                   dod = self.decoder_outputs_decode
                   self.beam_prob = dod.beam_search_decoder_output.scores

    def save(self,sess,save_path='model.ckpt',index=None):
        '''
        在TensorFlow里，保存模型的格式有两种：
        ckpt：训练模型后的保存，这里面会保存所有的训练参数，文件相对来讲比较大，可以用来进行模型的恢复和加载
        pb：用于模型的最后上线部署，这里面的线上部署指的是TensorFlow Serving进行模型的发布，一般发布成grpc形式
        的接口
        :param sess:
        :param save_path:
        :return:
        '''

        self.saver.save(sess,save_path=save_path+'-'+str(index))

    def load(self,sess,save_path='model.ckpt'):
        print('try load model from',save_path)
        self.saver.restore(sess,save_path)

    def init_optimizer(self):
        '''
        sgd,adadelta,adam,rmsprop,momentum
        :return:
        '''

        learning_rate = tf.train.polynomial_decay(
            self.learning_rate,
            self.global_step,
            self.decay_step,
            self.min_learning_rate,
            power=0.5
        )
        #初始化学习率的下降算法

        self.current_learning_rate = learning_rate

        #返回需要训练的参数的列表
        trainable_params = tf.trainable_variables()
        #设置优化器
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)

        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'momentum':
            self.opt = tf.train.MomentumOptimizer(
                learning_rate = learning_rate
            )
        elif self.optimizer.lower() == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate
            )

        gradients = tf.gradients(self.loss,trainable_params)
        clip_gradients,_ = tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )#对梯度下降进行裁剪

        #更新model
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                global_step=self.global_step)

        gradients = tf.gradients(self.loss_rewards,trainable_params)
        clip_gradients, _ =tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )
        self.updates_rewards = self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),
            global_step=self.global_step
        )

        #添加self.loss_add的update
        gradients = tf.gradients(self.add_loss,trainable_params)
        clip_gradients,_ = tf.clip_by_global_norm(
            gradients,self.max_gradient_norm
        )

        self.updates_add = self.opt.apply_gradients(
            zip(clip_gradients,trainable_params),
            global_step=self.global_step
        )

    def check_feeds(self,encoder_inputs,encoder_inputs_length,
                    decoder_inputs,decoder_inputs_length,decode):
        '''

        :param encoder_inputs:一个整型的二维矩阵，[batch_size,max_source_time_steps]
        :param encoder_inputs_length:[batch_size],每一个维度就是encoder句子的真实的长度
        :param decoder_inputs:一个整型的二维矩阵,[batch_size,max_target_time_steps]
        :param decoder_inputs_length:[batch_size]，每一个维度就是decoder句子的真实长度
        :param decode:
        :return:
        '''
        input_batch_size = encoder_inputs.shape[0]
        if input_batch_size != encoder_inputs_length.shape[0]:
            raise ValueError(
                'encoder_inputs 和 encoder_inputs_length的第一个维度必须一致'
                '这个维度是batch_size,%d!=%d'%(input_batch_size,encoder_inputs_length.shape[0]))

        if not decode:
            target_batch_size = decoder_inputs.shape[0]
            if target_batch_size != input_batch_size:
                raise ValueError(
                    'encoder_input和decoder_inputs的第一个维度'
                    '这个维度是batch_size,%d != %d' % (input_batch_size,target_batch_size)
                )
            if target_batch_size!=decoder_inputs_length.shape[0]:
                raise ValueError(
                    'encoder_inputs和decoder_inputs_length的第一个维度必须一致'
                    '这个维度是batch_size,%d!=%d'%(
                        target_batch_size,decoder_inputs_length.shape[0]
                    )
                )

        input_feed = {}

        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.encoder_inputs_length.name] = encoder_inputs_length

        if not decode:
            input_feed[self.decoder_inputs.name] = decoder_inputs
            input_feed[self.decoder_inputs_length.name] = decoder_inputs_length

        return input_feed

    def train(self,sess,encoder_inputs,encoder_inputs_length,decoder_inputs,decoder_inputs_length,rewards=None,return_lr=False,
              loss_only=False,add_loss=None):
        '''训练模型'''
        #输入
        input_feed = self.check_feeds(encoder_inputs,encoder_inputs_length,
                                      decoder_inputs,decoder_inputs_length,
                                      False)

        #设置dropout
        input_feed[self.keep_prob_placeholder.name] = self.keep_prob

        if loss_only:
            #输出
            return sess.run(self.loss,input_feed)

        if add_loss is not None:
            input_feed['add_loss:0'] = add_loss
            output_feed = [self.updates_add,self.add_loss,
                           self.current_learning_rate]
            _,cost,lr = sess.run(output_feed,input_feed)
            if return_lr:
                return cost,lr

        if rewards is not None:
            input_feed[self.rewards.name] = rewards
            output_feed = [
                self.updates_rewards,self.loss_rewards,
                self.current_learning_rate
            ]
            _,cost,lr = sess.run(output_feed,input_feed)
            if return_lr:
                return cost,lr
            return cost
        output_feed = [
            self.updates,self.loss,
            self.current_learning_rate
        ]

        _,cost,lr = sess.run(output_feed,feed_dict = input_feed)
        if return_lr:
            return cost,lr
        return cost

    def predict(self,sess,encoder_inputs,encoder_inputs_length,attention=False):
        input_feed = self.check_feeds(encoder_inputs,encoder_inputs_length,None,None,True)
        input_feed[self.keep_prob_placeholder.name] = 1.0

        if attention:
            assert not self.use_beamsearch_decode,'Attention 模式不能打开 BeamSearch'

            pred,atten = sess.run(
                [
                    self.decoder_pred_decode,
                    self.final_state.aligment_history.stack()
                ],input_feed)#如果使用attention，就用历史来做预测。
            return pred,atten

        if self.use_beamsearch_decode:
            pred,beam_prob = sess.run(
                [
                    self.decoder_pred_decode,self.beam_prob #beamsearch
                ],input_feed)
            beam_prob = np.mean(beam_prob,axis=1)

            pred = pred[0]

            return pred

        pred, = sess.run([
            self.decoder_pred_decode
        ],input_feed)
        return pred






























































