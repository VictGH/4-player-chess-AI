# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:50:10 2020

@author: Victor
"""
import chessgame4
import sys
sys.path.append('..')
#from utils import *

import tensorflow as tf

class OthelloNNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Renaming functions 
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        BatchNormalization = tf.compat.v1.layers.batch_normalization
        #Dropout = tf.compat.v1.layers.Dropout
        Dense = tf.compat.v1.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            self.input_boards = tf.compat.v1.placeholder(tf.float32, shape=[None, self.board_x, self.board_y,2])    # s: batch_size x board_x x board_y
            self.dropout = tf.compat.v1.placeholder(tf.float32)
            self.isTraining = tf.compat.v1.placeholder(tf.bool, name="is_training")

            self.x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 2])                    # batch_size  x board_x x board_y x 1
            self.h_conv1 = Relu(BatchNormalization(self.conv2d(self.x_image, args['num_channels'], 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            self.h_conv2 = Relu(BatchNormalization(self.conv2d(self.h_conv1, args['num_channels'], 'same'), axis=3, training=self.isTraining))     # batch_size  x board_x x board_y x num_channels
            self.h_conv3 = Relu(BatchNormalization(self.conv2d(self.h_conv2, args['num_channels'], 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-2) x (board_y-2) x num_channels
            self.h_conv4 = Relu(BatchNormalization(self.conv2d(self.h_conv3, args['num_channels'], 'valid'), axis=3, training=self.isTraining))    # batch_size  x (board_x-4) x (board_y-4) x num_channels
            self.h_conv4_flat = tf.reshape(self.h_conv4, [-1, args['num_channels']*(self.board_x-4)*(self.board_y-4)])
            self.s_fc1 = self.Dropout(Relu(BatchNormalization(Dense(self.h_conv4_flat, 1024, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout) # batch_size x 1024
            self.s_fc2 = self.Dropout(Relu(BatchNormalization(Dense(self.s_fc1, 512, use_bias=False), axis=1, training=self.isTraining)), rate=self.dropout)         # batch_size x 512
            self.pi = Dense(self.s_fc2, self.action_size)                                                        # batch_size x self.action_size
            self.prob = tf.nn.softmax(self.pi)
            self.v = Tanh(Dense(self.s_fc2, 1))                                                               # batch_size x 1

            self.calculate_loss()
            """
            print(self.pi)
            print(self.prob)
            print(self.v)
            """
          
    def conv2d(self, x, out_channels, padding):
      return tf.compat.v1.layers.Conv2D( out_channels, kernel_size=3, padding=padding, use_bias=False)(x)
    
    def Dropout(self, x, rate):
      return tf.compat.v1.layers.Dropout(rate)(x)

    def calculate_loss(self):
        self.target_pis = tf.compat.v1.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.compat.v1.placeholder(tf.float32, shape=[None])
        self.loss_pi =  tf.compat.v1.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.compat.v1.losses.mean_squared_error(self.target_vs, tf.reshape(self.v, shape=[-1,]))
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = tf.compat.v1.train.AdamOptimizer(self.args['lr']).minimize(self.total_loss)

class ResNet():
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        BatchNormalization = tf.compat.v1.layers.batch_normalization
        Relu = tf.nn.relu
        Tanh = tf.nn.tanh
        Dense = tf.compat.v1.layers.dense

        # Neural Net
        self.graph = tf.Graph()
        with self.graph.as_default(): 
            self.input_boards = tf.compat.v1.placeholder(tf.float32, shape=[None, self.board_x, self.board_y,10])    # s: batch_size x board_x x board_y
            self.dropout = tf.compat.v1.placeholder(tf.float32)
            self.isTraining = tf.compat.v1.placeholder(tf.bool, name="is_training")

            x_image = tf.reshape(self.input_boards, [-1, self.board_x, self.board_y, 10])                    # batch_size  x board_x x board_y x 1
            x_image = self.conv2d(x_image, args['num_channels'],kernel_size=(3, 3), name='conv',padding='same')
            x_image = BatchNormalization(x_image, axis=3, name='conv_bn', training=self.isTraining)
            x_image = Relu(x_image)

            residual_tower = self.residual_block(inputLayer=x_image, kernel_size=3, filters=args['num_channels'], stage=1, block='a')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=2, block='b')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=3, block='c')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=4, block='d')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=5, block='e')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=6, block='g')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=7, block='h')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=8, block='i')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=9, block='j')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=10, block='k')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=11, block='m')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=12, block='n')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=13, block='o')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=14, block='p')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=15, block='q')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=16, block='r')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=17, block='s')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=18, block='t')
            residual_tower = self.residual_block(inputLayer=residual_tower, kernel_size=3, filters=args['num_channels'], stage=19, block='u')

            policy = self.conv2d(residual_tower, 2,kernel_size=(1, 1), name='pi',padding='same')
            policy = BatchNormalization(policy, axis=3, name='bn_pi', training=self.isTraining)
            policy = Relu(policy)
            policy = tf.compat.v1.layers.flatten(policy, name='p_flatten')
            self.pi = Dense(policy, self.action_size)
            self.prob = tf.nn.softmax(self.pi)

            value = self.conv2d(residual_tower, 1,kernel_size=(1, 1),name='v',padding='same')
            value = BatchNormalization(value, axis=3, name='bn_v', training=self.isTraining)
            value = Relu(value)
            value = tf.compat.v1.layers.flatten(value, name='v_flatten')
            value = Dense(value, units=256)
            value = Relu(value)
            value = Dense(value, 4)
            self.v = Tanh(value)*9
                                                              
            self.calculate_loss()

    def residual_block(self,inputLayer, filters,kernel_size,stage,block):
        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'
        BatchNormalization = tf.compat.v1.layers.batch_normalization
        Relu = tf.nn.relu
        shortcut = inputLayer
        residual_layer = self.conv2d(inputLayer, filters,kernel_size=(kernel_size, kernel_size), name=conv_name+'2a',padding='same')
        residual_layer = BatchNormalization(residual_layer, axis=3, name=bn_name+'2a', training=self.isTraining)
        residual_layer = Relu(residual_layer)
        residual_layer = self.conv2d(residual_layer, filters,kernel_size=(kernel_size, kernel_size),name=conv_name+'2b',padding='same')
        residual_layer = BatchNormalization(residual_layer, axis=3, name=bn_name+'2b', training=self.isTraining)
        add_shortcut = tf.math.add(residual_layer, shortcut)
        residual_result = Relu(add_shortcut)
        
        return residual_result

    def calculate_loss(self):
        self.target_pis = tf.compat.v1.placeholder(tf.float32, shape=[None, self.action_size])
        self.target_vs = tf.compat.v1.placeholder(tf.float32, shape=[None,4])
        self.loss_pi =  tf.compat.v1.losses.softmax_cross_entropy(self.target_pis, self.pi)
        self.loss_v = tf.compat.v1.losses.mean_squared_error(self.target_vs, self.v)
        self.total_loss = self.loss_pi + self.loss_v
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step =  tf.compat.v1.train.AdamOptimizer(self.args['lr']).minimize(self.total_loss)
    
    def conv2d(self, x, out_channels,kernel_size, padding,name):
      return tf.compat.v1.layers.Conv2D( out_channels, kernel_size=kernel_size, padding=padding, use_bias=False,name = name)(x)




if __name__ == "__main__":
    args  = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})
    
    
  
    game = chessgame.GameBoard()
    net = OthelloNNet(game,args)           
            
