# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 14:50:09 2020

@author: Victor
"""

import logging
import math
import chessgame4
import nnet4
import numpy as np
from utils4 import *
import wrapper4
import sys
import time
EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard,story,player,points, temp=1):
        #with open('/content/drive/MyDrive/chess/output.txt','a') as out:
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}  
        self.Ps = {}  

        self.Es = {}  
        self.Vs = {} 
        
        epsilon = 0.25
        dirich = [epsilon*j for j in np.random.dirichlet([0.03 for i in range(self.game.getActionSize())])]    
        valids = [1 if j in self.game.getValidMoves(canonicalBoard,player) else 0 for j in range(self.game.getActionSize())]
        dirich = [a * b for a, b in zip(dirich, valids)]
        dirich = dirich/(np.sum(dirich))
        t = np.copy(np.array(canonicalBoard))
        s = (tuple(t.flatten()),story[t.tobytes()],player)  
        self.Ps[s], v1 = self.nnet.predict(canonicalBoard,story,player,points)
        valids = self.game.getValidMoves(canonicalBoard, player)
        self.Ps[s] = self.Ps[s] * [1 if i in valids else 0 for i in range(self.game.getActionSize())]  # masking invalid moves
        sum_Ps_s = np.sum(self.Ps[s])
        if sum_Ps_s > 0:
            self.Ps[s] /= sum_Ps_s
        self.Vs[s] = valids
        self.Ns[s] = 0
        self.Ps[s] = [(1-epsilon)*self.Ps[s][i]+epsilon*dirich[i] for i in range(self.game.getActionSize())]
        
        #print('caballo')
        #self.game.visual(canonicalBoard)
        #time.sleep(7)
        for i in range(self.args.numMCTSSims):
            #print(i)
            #print('iiiii')
            gstory = story.copy()
            soints = points.copy()
            a = np.copy(np.array(canonicalBoard))
            depth = 0
            self.search(a,depth,gstory,player,soints)
        t = np.copy(np.array(canonicalBoard))
        s = (tuple(t.flatten()),story[(t).tobytes()],player)
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            #print('temp =0')
            #print(counts)
            #print('counts')
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        
        #print('temp =1')
        return probs

    def search(self, canonicalBoard,d,story,player,points):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        #self.game.visual(canonicalBoard)
        #print(d)
        #print('----------------------')
        d+=1
        t = np.copy(np.array(canonicalBoard))
        s = (tuple(t.flatten()),story[t.tobytes()],player)
        #print(story[t.tobytes()])
       
        if s not in self.Es:
            #print('not E')
            self.Es[s] = self.game.getGameEnded(canonicalBoard,story,points)[0]
        if self.game.getGameEnded(canonicalBoard,story,points)[1] != 0:
            # terminal node
            #print('acaboo')
            return self.Es[s]

        if s not in self.Ps:
            #print('not P')
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard,story,player,points)
            #print(v,s)
            #print('v,s')
            
            valids = self.game.getValidMoves(canonicalBoard, player)
            self.Ps[s] = self.Ps[s] * [1 if i in valids else 0 for i in range(self.game.getActionSize())]  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                sys.exit("Error message")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        #print('pick action')
        # pick the action with the highest upper confidence bound
        for a in valids:
           
            if (s, a) in self.Qsa:
                #print('qsa')
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                        1 + self.Nsa[(s, a)])
                #print(self.Ps[s][a])
                #print(self.Ns[s])
                #print(self.Nsa[(s,a)])
                #print(s,a)
                #print(self.Qsa[(s, a)])
                #print(u)
                #print('long check')
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
                #print('else')
                #print(self.Ps[s][a])
                #print(self.Ns[s])
                #print(s,a)
            #print(cur_best)
            #print('cur_best')
            if u > cur_best:
                #print(u)
                #print('cur_best2')
                cur_best = u
                best_act = a
        #print('finish best action')
        a = best_act
        #print(a)
        #print('aaaaa')

        if a not in valids:
            log.error(f'Action {a} is not valid!')
            log.debug(f'valids = {valids}')
            print(len(valids),'len valids')
            print(canonicalBoard)
            print(a,'a')
            print(valids)
           
            assert a in valids
        next_s, player,story,points = self.game.getNextState(canonicalBoard, player, a,story,points)
       
       
        #print('buscando')
        v = self.search(next_s,d,story,player,points)
        player = (player-1)%4
        if (s, a) in self.Qsa:
            #print(v)
            #print('v qsa')
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] -sum(v)+2*v[player]) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
            #print(self.Qsa[(s, a)])
            #print('qsa')
        else:
            #print('else2')
            self.Qsa[(s, a)] = -sum(v)+2*v[player]
            #print(v)
            #print('v')
            #print(player)
            #print('player')
            #print(-sum(v))
            #print('-sum(v)')
            #print(v[player])
            #print('v[player]')
            #print(s,a)
            #print('qqq')
            #print(self.Qsa[(s, a)])
            #print('qsa')
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
    
    
if __name__ == "__main__":
    
    
    game = chessgame.GameBoard()
    
    game.addpiece("wknight1", 1, 3,0)
    game.addpiece("wknight2", 1, 3,1)
    game.addpiece("wknight3", 1, 3,2)
    game.addpiece("wknight4", 1, 3,3)
    game.addpiece("bknight1", -1, 0,0)
    game.addpiece("bknight2", -1, 0,1)
    game.addpiece("bknight3", -1, 0,2)
    game.addpiece("bknight4", -1, 0,3)
    args2  = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})
    args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 300,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})
    #net = nnet.OthelloNNet(game,args2) 
    wraper = wrapper.NNetWrapper(game)
    mcsquare = MCTS(game,wraper,args)
    r = mcsquare.getActionProb(game.board)
    root.mainloop()