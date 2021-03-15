# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 14:57:28 2021

@author: Victor
"""

import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
import tkinter as tk

import math
import chessgame4
import nnet4

from utils4 import *
import wrapper4

import numpy as np
from tqdm import tqdm

from Arena4 import Arena
from MCTS4 import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 0
        episodeStep = 0
        story = {}
        story[np.array(board).tobytes()]=1
        
        points = np.array([0,0,0,0])
        while True:
            episodeStep += 1
            
            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(board,story,self.curPlayer,points, temp=temp)
            z=story[(np.copy(np.array(board))).tobytes()]
            z = z*np.ones((board.shape[1:]),dtype =int)
            y = self.curPlayer*np.ones((board.shape[1:]),dtype =int)
            q = [0,0,0,0]
            for i in range(4):
                q[i] = points[i]*np.ones((board.shape[1:]),dtype =int)
            trainExamples.append([board,z,y, pi,q, None])
            action = np.argmax(pi)
            print(action)
            print('action')
            board, self.curPlayer,story,points = self.game.getNextState(board, self.curPlayer, action,story,points)
            print(board[0]+board[1]*2+board[2]*3+board[3]*4)
            print(board.shape[1:])

            r1,r2 = self.game.getGameEnded(board,story,points)
            
            if r2 != 0:
                print(board)
                return [(np.stack([x[0][0],x[0][1],x[0][2],x[0][3],x[1],x[2],q[0],q[1],q[2],q[3]],axis = 2), x[3], points) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x,y,z,t: np.argmax(nmcts.getActionProb(x,y,z,t,temp=0)),
                          lambda x,y,z,t: np.argmax(pmcts.getActionProb(x,y,z,t, temp=0)),
                          lambda x,y,z,t: np.argmax(pmcts.getActionProb(x,y,z,t,temp=0)),
                          lambda x,y,z,t: np.argmax(pmcts.getActionProb(x,y,z,t,temp=0)),self.game)
            totalpoints = arena.playGames(self.args.arenaCompare)

            print(totalpoints)
            print('totalpoints')
            if sum(totalpoints)==0 or totalpoints[0]/sum(totalpoints) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self,iteration):
        modelFile = os.path.join(self.args.load_folder_file[0], self.getCheckpointFile(iteration))
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = "y"#input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')
            print('skippppppping')
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True