import tensorflow as tf

class CNN:

    def __init__(self, height, width, depth, opt):
        self.input = tf.placeholder(tf.float32, [None, height, width, depth], name='X_placeholder')
        self.target = tf.placeholder(tf.float32, [None, 3], name='y_placeholder')
        self.height = height
        self.width = width
        self.depth = depth
        self.opt = opt

    def build_graph(self, reuse, train_mode):
        self.train_mode = train_mode
        with tf.variable_scope('cnn', reuse=reuse):
            self.output = self.input

            # filter_nums = [64, 128, 256, 128, 64, 3]
            # for i, filter_num in enumerate(filter_nums):
            #     self.output = self.conv_layer(self.output, 'conv_%s' % i, filter_num)

            # with tf.variable_scope('conv_final'):
            #     self.output = self.conv(self.output, 3, 1)

            # self.output = tf.reduce_sum(self.output, axis=[1, 2])

            # self.shape = tf.shape(self.output)
            # self.output = tf.reshape(self.output, [self.shape[0], self.height * self.width * self.depth])
            #
            # layer_sizes = [256, 128, 64, 32]
            # for i, layer_size in enumerate(layer_sizes):
            #     self.output = tf.layers.dense(self.output, units=layer_size, activation=tf.nn.elu, use_bias=True, name='fc_%s' % i)
            #     if self.train_mode:
            #         self.output = tf.nn.dropout(self.output, keep_prob=0.95)
            #
            # self.output = tf.layers.dense(self.output, units=3, activation=None, use_bias=True, name="fc_last")

            #  VGG-16 https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
            # filter_num = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
            #
            # self.conv1_1 = self.conv_layer(self.input, 'conv1_1', filter_num[0])
            # self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2', filter_num[1])
            # self.pool1 = self.avg_pool(self.conv1_2, 'pool1')
            #
            # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", filter_num[2])
            # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", filter_num[3])
            # self.pool2 = self.avg_pool(self.conv2_2, 'pool2')
            #
            # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", filter_num[4])
            # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", filter_num[5])
            # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", filter_num[6])
            # self.pool3 = self.avg_pool(self.conv3_3, 'pool3')
            #
            # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", filter_num[7])
            # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", filter_num[8])
            # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", filter_num[9])
            # self.pool4 = self.avg_pool(self.conv4_3, 'pool4')
            #
            # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", filter_num[10])
            # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", filter_num[11])
            # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", filter_num[12])
            # self.pool5 = self.avg_pool(self.conv5_3, 'pool5')
            #
            # self.shape = tf.shape(self.pool5)
            # fc_input = tf.reshape(self.pool5, [self.shape[0], 131072])
            #
            # with tf.variable_scope('fc6'):
            #     self.fc6 = tf.layers.dense(fc_input, units=128, activation=tf.nn.elu, use_bias=True, name="fc6")
            # if train_mode:
            #     self.fc6 = tf.nn.dropout(self.fc6, keep_prob=0.95, name='dropout6')
            #
            # with tf.variable_scope('fc7'):
            #     self.fc7 = tf.layers.dense(self.fc6, units=128, activation=tf.nn.elu, use_bias=True, name="fc7")
            # if train_mode:
            #     self.fc7 = tf.nn.dropout(self.fc7, keep_prob=0.95, name='dropout7')
            #
            # with tf.variable_scope('fc8'):
            #     self.output = tf.layers.dense(self.fc7, units=3, activation=None, use_bias=True, name="fc8")
            #     # self.output = tf.layers.dense(self.fc7, units=3, activation=tf.nn.sigmoid, use_bias=True, name="fc8")

            filter_nums = [16, 32, 64, 128]
            for i, filter_num in enumerate(filter_nums):
                self.output = tf.layers.conv2d(self.output, filter_num, kernel_size=2, strides=(2, 2), padding='valid', kernel_initializer=tf.contrib.layers.xavier_initializer())
                self.output = tf.nn.elu(self.output)
                self.output = self.avg_pool(self.output, 'pool_%s' % i)

            self.output = tf.reshape(self.output, [tf.shape(self.output)[0], filter_nums[-1]])
            print(self.output)
            self.output = tf.layers.dense(self.output, units=3, kernel_initializer=tf.contrib.layers.xavier_initializer())

            self.color = self.output

            self.loss = tf.reduce_mean(tf.square(tf.subtract(self.target, self.output)))
            # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=target))

            vars = [var for var in tf.trainable_variables()]
            self.optimizer = tf.train.AdamOptimizer(self.opt.lr, self.opt.beta1)
            self.grads_and_vars = self.optimizer.compute_gradients(self.loss, var_list=vars)
            self.train = self.optimizer.apply_gradients(self.grads_and_vars)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv(self, batch_input, out_channels, stride):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [1, 1, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [0, 0], [0, 0], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

    def conv_layer(self, bottom, name, filter_num):
        with tf.variable_scope(name):
            # conv = self.conv(bottom, filter_num, 1)
            conv = tf.layers.conv2d(bottom, filter_num, 1, (1, 1), padding='valid', kernel_initializer=tf.random_normal_initializer(0, 0.02))
            relu = tf.nn.elu(conv)
        return relu