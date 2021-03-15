# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:51:51 2021

@author: Victor
"""

import logging

import chessgame4
import nnet4
import numpy as np
from utils4 import *
import wrapper4
from tqdm import tqdm
import MCTS4

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player0, player1,player2,player3, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player0 = player0
        self.player1 = player1
        self.player2 = player2
        self.player3 = player3
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.
        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player0, self.player1, self.player2,self.player3]
        curPlayer = 0
        points = [0,0,0,0]
        board = self.game.getInitBoard()
        it = 0
        story = {(np.copy(np.array(board))).tobytes():1}
        while self.game.getGameEnded(board,story,points)[1] == 0:
         
        
            #print(board)
            #print(it)
            #print('curr'+str(curPlayer))
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)
            action = players[curPlayer](board,story,curPlayer,points)
           
            valids = self.game.getValidMoves(board, curPlayer)
            #print(np.where(valids==1.))
           

            if action not in valids:

                log.error(f'Action {action} is not valid!')
                print(board)
                print('board')
                print(curPlayer)
                print('curPlayer')
                print(action)
                print('action')
                log.debug(f'valids = {valids}')
                assert action in valids
            board, curPlayer,story,points = self.game.getNextState(board, curPlayer, action,story,points)
            print(board[0]+board[1]*2+board[2]*3+board[3]*4)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, story)[0]))
            self.display(board)
    
        print(board)
        print('fin')
        
        return self.game.getGameEnded(board,story,points)[0]

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        listin = []
        num = int(num / 4)
        totalpoints = np.array([0,0,0,0])
    
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            totalpoints+=gameResult
            listin.append(gameResult)

        self.player0, self.player1,self.player2,self.player3 = self.player3, self.player0,self.player1,self.player2
        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            totalpoints[0]+=gameResult[1]
            totalpoints[1]+=gameResult[2]
            totalpoints[2]+=gameResult[3]
            totalpoints[3]+=gameResult[0]
            listin.append(gameResult)

        self.player0, self.player1,self.player2,self.player3 = self.player3, self.player0,self.player1,self.player2
        for _ in tqdm(range(num), desc="Arena.playGames (3)"):
            gameResult = self.playGame(verbose=verbose)
            totalpoints[0]+=gameResult[2]
            totalpoints[1]+=gameResult[3]
            totalpoints[2]+=gameResult[0]
            totalpoints[3]+=gameResult[1]
            listin.append(gameResult)

        self.player0, self.player1,self.player2,self.player3 = self.player3, self.player0,self.player1,self.player2
        for _ in tqdm(range(num), desc="Arena.playGames (4)"):
            gameResult = self.playGame(verbose=verbose)
            totalpoints[0]+=gameResult[3]
            totalpoints[1]+=gameResult[0]
            totalpoints[2]+=gameResult[1]
            totalpoints[3]+=gameResult[2]
            listin.append(gameResult)
            
        print(listin)
        return totalpoints
    
if __name__ == "__main__":
    
    
    game = chessgame.GameBoard()
    
    game.addpiece("wknight1", 1, 3,0)
    game.addpiece("wknight2", 1, 3,1)
    game.addpiece("wknight3",1, 3,2)
    game.addpiece("wknight4",1, 3,3)
    game.addpiece("bknight1",-1, 0,0)
    game.addpiece("bknight2",-1, 0,1)
    game.addpiece("bknight3",-1, 0,2)
    game.addpiece("bknight4",-1, 0,3)
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
    wraper1 = wrapper.NNetWrapper(game)
    wraper2 = wrapper.NNetWrapper(game)
    pmcts = MCTS.MCTS(game, wraper2, args)
    nmcts = MCTS.MCTS(game, wraper1, args)
    arena = Arena(lambda x,y,z,t: np.argmax(pmcts.getActionProb(x,y,z,t, temp=0)),
                          lambda x,y,z,t: np.argmax(nmcts.getActionProb(x,y,z,t, temp=0)), game)
    
    patata = arena.playGame()
    print('end')
    root.mainloop()