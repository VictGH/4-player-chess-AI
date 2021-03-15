# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:22:35 2021

@author: Victor
"""

import logging

#import coloredlogs

import logging
import math
import chessgame4

import numpy as np
from utils4 import *
import wrapper4
import sys
from Coach4 import Coach


from utils import *

log = logging.getLogger(__name__)

#coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.2875,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 15,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': '/content/drive/MyDrive/chess/',
    'load_model': True,
    'load_folder_file': ('/content/drive/MyDrive/chess/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    #log.info('Loading %s...', Game.__name__)
   

    #log.info('Loading %s...', nn.__name__)
    nnet = wrapper4.NNetWrapper(game)
    
    if True:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
        print('loaded model..........')
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(game, nnet, args)

    if False:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples(15)

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    
   
    game = chessgame4.GameBoard()
    
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
    'epochs': 40,
    'batch_size': 64,
    'num_channels': 512,
})
  
    main()
    
    
    
