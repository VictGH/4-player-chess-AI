# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:26:34 2020

@author: Victor
"""

import os
import sys
import time

import numpy as np
from tqdm import tqdm
import chessgame4
sys.path.append('../../')
from utils4 import *

import tensorflow as tf

from nnet4 import ResNet as rnet


args = dotdict({
    'lr': 0.0001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})


class NNetWrapper():
    def __init__(self, game):
        self.nnet = rnet(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session(graph=self.nnet.graph)
        self.saver = None
        with tf.compat.v1.Session() as temp_sess:
            temp_sess.run(tf.compat.v1.global_variables_initializer())
        self.sess.run(tf.compat.v1.variables_initializer(self.nnet.graph.get_collection('variables')))

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / args.batch_size)

            # self.sess.run(tf.local_variables_initializer())
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                input_dict = {self.nnet.input_boards: boards, self.nnet.target_pis: pis, self.nnet.target_vs: vs,
                              self.nnet.dropout: args.dropout, self.nnet.isTraining: True}

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run([self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict)
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

    def predict(self, board,story,player,points):
        """
        board: np array with board
        """
        # timing
        #start = time.time()

        # preparing input
        t = np.copy(np.array(board))
        t = story[(t).tobytes()]
        t = t*np.ones((board.shape[1:]),dtype =int)
        y = player*np.ones((board.shape[1:]),dtype =int)
        q = [0,0,0,0]
        for i in range(4):
            q[i] = points[i]*np.ones((board.shape[1:]),dtype =int)
        board2  = np.stack([board[0],board[1],board[2],board[3],t,y,q[0],q[1],q[2],q[3]], axis=2)
        board2 = np.copy(board2[np.newaxis, :, :,:])
        
        # run
        prob, v = self.sess.run([self.nnet.prob, self.nnet.v],
                                feed_dict={self.nnet.input_boards: board2, self.nnet.dropout: 0,
                                           self.nnet.isTraining: False})
       
        
        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return prob[0], v[0]
    
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        if self.saver == None:
            self.saver = tf.compat.v1.train.Saver(self.nnet.graph.get_collection('variables'))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath + '.meta'):
            raise ("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.compat.v1.train.Saver()
            self.saver.restore(self.sess, filepath)

if __name__ == "__main__":
    
   
    board = chessgame.GameBoard()
    
    board.addpiece("wknight1", 1, 3,0)
    board.addpiece("wknight2", 1, 3,1)
    board.addpiece("wknight3", 1, 3,2)
    board.addpiece("wknight4", 1, 3,3)
    board.addpiece("bknight1", -1, 0,0)
    board.addpiece("bknight2", -1, 0,1)
    board.addpiece("bknight3", -1, 0,2)
    board.addpiece("bknight4", -1, 0,3)
    
    wraper = NNetWrapper(board)
    print(wraper.predict(board.board))
    root.mainloop()