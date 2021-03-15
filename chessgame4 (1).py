import tkinter as tk
import numpy as np
from copy import copy, deepcopy

class GameBoard():
    def __init__(self,  rows=5, columns=5, size=80, color1="white", color2="grey"):
        '''size is the size of a square, in pixels'''

        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}
        self.blue = []
        self.pick = [0,(0,0)] #Equal to 1 if a piece is chosen and its position
        self.pcolors = {}
        self.turn = 1
        self.white = []
        self.black = []
        self.board = np.zeros((self.rows, self.columns),dtype = int)
        canvas_width = columns * size
        canvas_height = rows * size

        
        # this binding will cause a refresh if the user interactively
        # changes the window size
        
    def addpiece(self, name, pcolor, row=0, column=0):
        '''Add a piece to the playing board'''
        self.pcolors[name] = pcolor
        self.placepiece(name, row, column)
    
    def deletepiece(self, name):
        '''Delete a piece from the playing board'''
    
        del self.pcolors[name]
        del self.pieces[name]
        self.white =  [v for k,v in self.pieces.items() if self.pcolors[k]==1]
        self.black =  [v for k,v in self.pieces.items() if self.pcolors[k]==-1]
        self.toBoard()
    

    def placepiece(self, name, row, column):
        '''Place a piece at the given row/column'''
        self.pieces[name] = (row, column)
        x0 = (column * self.size) + int(self.size/2)
        y0 = (row * self.size) + int(self.size/2)
        
        self.white =  [v for k,v in self.pieces.items() if self.pcolors[k]==1]
        self.black =  [v for k,v in self.pieces.items() if self.pcolors[k]==-1]
        self.toBoard()
        

    def select(self,event):
       
       
        x1 = int(event.x/self.size)*self.size
        y1 = int(event.y/self.size)*self.size
        x2 = x1 + self.size
        y2 = y1 + self.size
        
        
        self.available(x1,y1)
        self.move(x1,y1)
        self.white =  [v for k,v in self.pieces.items() if self.pcolors[k]==1]
        self.black =  [v for k,v in self.pieces.items() if self.pcolors[k]==-1]
        self.toBoard()
        
    def available(self,x,y):
        
        position_x = int(x/self.size)
        position_y = int(y/self.size)
        piece_pos_player = [v for k,v in self.pieces.items() if self.pcolors[k]==self.turn]
        
        if self.pick[0] ==0:
            self.blue=[]
        
        #0
        if self.pick[0] ==0 and position_x+1<self.rows and position_y-2>=0 and (position_y,position_x) in piece_pos_player and (position_y-2,position_x+1) not in piece_pos_player:
            x0 = x+self.size
            y0 = y-2*self.size
            self.blue.append((x0,y0))
        #1
        if self.pick[0] ==0 and position_x+2<self.rows and position_y-1>=0 and(position_y,position_x) in piece_pos_player and (position_y-1,position_x+2) not in piece_pos_player:
            x0 = x+2*self.size
            y0 = y-self.size
            self.blue.append((x0,y0))
            
        #2
        if self.pick[0] ==0 and position_x+2<self.rows and position_y+1<self.columns and(position_y,position_x) in piece_pos_player and (position_y+1,position_x+2) not in piece_pos_player:
            x0 = x+2*self.size
            y0 = y+self.size
            self.blue.append((x0,y0))
            
        #3
        if self.pick[0] ==0 and position_x+1<self.rows and position_y+2<self.columns and(position_y,position_x) in piece_pos_player and (position_y+2,position_x+1) not in piece_pos_player:
            x0 = x+self.size
            y0 = y+2*self.size
            self.blue.append((x0,y0))
            
        #4
        if self.pick[0] ==0 and position_x-1>=0 and position_y+2<self.columns and(position_y,position_x) in piece_pos_player and (position_y+2,position_x-1) not in piece_pos_player:
            x0 = x-self.size
            y0 = y+2*self.size
            self.blue.append((x0,y0))
            
        #5
        if self.pick[0] ==0 and position_x-2>=0 and position_y+1<self.columns and (position_y,position_x) in piece_pos_player and (position_y+1,position_x-2) not in piece_pos_player:
            x0 = x-2*self.size
            y0 = y+self.size
            self.blue.append((x0,y0))
            
        #6
        if self.pick[0] ==0 and position_x-2>=0 and position_y-1>=0 and(position_y,position_x) in piece_pos_player and (position_y-1,position_x-2) not in piece_pos_player:
            x0 = x-2*self.size
            y0 = y-self.size
            self.blue.append((x0,y0))
            
        #7
        if self.pick[0] ==0 and position_x-1>=0 and position_y-2>=0 and(position_y,position_x) in piece_pos_player and (position_y-2,position_x-1) not in piece_pos_player:
            x0 = x-self.size
            y0 = y-2*self.size
            self.blue.append((x0,y0))
            
        
        if (position_y,position_x) in piece_pos_player and self.pick[0]==0:
            self.pick = [1,(position_y,position_x)]
        elif self.pick[0]==1:
            self.pick[0] = 2
            
        
    
    def move(self,x,y):
        rev_dict = {v: k for k, v in self.pieces.items()}
        position_x2 = int(x/self.size)
        position_y2 = int(y/self.size)
        piece_pos_opponent = [v for k,v in self.pieces.items() if self.pcolors[k]==(-self.turn)]
        piece_pos_player = [v for k,v in self.pieces.items() if self.pcolors[k]==self.turn]
        if self.pick[0]==2 and (position_x2*self.size,position_y2*self.size) in self.blue:
            
            piece_name = rev_dict[(self.pick[1][0],self.pick[1][1])]
            self.placepiece(piece_name, position_y2,position_x2)
            self.pick[0] = 0
            self.turn = -self.turn
           
            if (position_y2,position_x2) in piece_pos_opponent:
                self.deletepiece(rev_dict[(position_y2,position_x2)])
        elif self.pick[0]==2 and (position_y2,position_x2) in piece_pos_player:
            self.pick[0]=0
            self.available(x,y)
        
        elif self.pick[0] ==2:
            self.pick[0]=0
        
        
    def getBoardSize(self):
        return (self.rows,self.columns)
    
     
    def getActionSize(self):
        return 200
            
    def toBoard(self):
        for j,k in self.white:
            self.board[j][k]=1
        for j,k in self.black:
            self.board[j][k]=-1
            
    def getGameEnded(self, board,story,points):
        if story[np.array(board).tobytes()]>=3:
            return points,1
        if len(story) >= 100:
            return points,1
        if np.count_nonzero(board[0] == 1) ==0:
            return points,1
        elif np.count_nonzero(board[1] == 1) ==0:
            return points,1
        elif np.count_nonzero(board[2] == 1) ==0:
            return points,1
        elif np.count_nonzero(board[3] == 1) ==0:
            return points,1
        elif np.count_nonzero(board[3] == 1)==1 and np.count_nonzero(board[2] == 1)==1 and np.count_nonzero(board[1] == 1)==1 and np.count_nonzero(board[0] == 1)==1:
            return points,1
        else:
            return points,0
    def getValidMoves(self,board,player):
        moves = []
        n = 5
       
        t = list(zip(*np.where(board[player] == 1)))
        s = list(zip(*np.where(board[player] == 1)))
        t.append((0,0))
        t.append((0,4))
        t.append((4,0))
        t.append((4,4))
  
        for z in s:
            if z[1]+1<n and z[0]-2>=0 and (z[0]-2,z[1]+1) not in t:
                moves.append(0*n**2+z[1]*n+z[0])
            if z[1]+2<n and z[0]-1>=0 and (z[0]-1,z[1]+2) not in t:
                moves.append(1*n**2+z[1]*n+z[0])
            if z[1]+2<n and z[0]+1<n and (z[0]+1,z[1]+2) not in t:
                moves.append(2*n**2+z[1]*n+z[0])
            if z[1]+1<n and z[0]+2<n and (z[0]+2,z[1]+1) not in t:
                moves.append(3*n**2+z[1]*n+z[0])
            if z[1]-1>=0 and z[0]+2<n and (z[0]+2,z[1]-1) not in t:
                moves.append(4*n**2+z[1]*n+z[0])
            if z[1]-2>=0 and z[0]+1<n and (z[0]+1,z[1]-2) not in t:
                moves.append(5*n**2+z[1]*n+z[0])
            if z[1]-2>=0 and z[0]-1>=0 and (z[0]-1,z[1]-2) not in t:
                moves.append(6*n**2+z[1]*n+z[0])
            if z[1]-1>=0 and z[0]-2>=0 and (z[0]-2,z[1]-1) not in t:
                moves.append(7*n**2+z[1]*n+z[0])
        
        return moves

      
    def getInitBoard(self):
        b = np.zeros((4,self.rows,self.columns), dtype=int)
        b[0][3][2] = 1
        b[0][4][1] = 1
        b[0][4][3] = 1
        
        b[1][1][4] = 1
        b[1][2][3] = 1
        b[1][3][4] = 1
        
        b[2][0][1] = 1
        b[2][0][3] = 1
        b[2][1][2] = 1
        
        b[3][1][0] = 1
        b[3][3][0] = 1
        b[3][2][1] = 1
        
        return b
        
    def getCanonicalForm(self,board, curPlayer):
        return np.copy(board)*curPlayer
    
    def visual(self,board):
        print(board[0]+board[1]*2+board[2]*3+board[3]*4)
    
    def getNextState(self,board, player, move,story,points):
        n= 5 
        initial_row = move%n
        move = int(move/n)
        initial_col = move%n
        move =  int(move/n)
        if move ==0:
            final_row,final_col = (initial_row-2,initial_col+1)
        elif move ==1:
            final_row,final_col = (initial_row-1,initial_col+2)
        elif move ==2:
            final_row,final_col = (initial_row+1,initial_col+2)
        elif move ==3:
            final_row,final_col = (initial_row+2,initial_col+1)
        elif move ==4:
            final_row,final_col = (initial_row+2,initial_col-1)
        elif move ==5:
            final_row,final_col = (initial_row+1,initial_col-2)
        elif move ==6:
            final_row,final_col = (initial_row-1,initial_col-2)
        elif move ==7:
            final_row,final_col = (initial_row-2,initial_col-1)
        p = np.copy(points) 
        y =  np.copy(board)
        y[player][initial_row][initial_col]=0
        y[player][final_row][final_col]=1
        for i in range(4):
            if i==player:
                continue
            if y[i][final_row][final_col] ==1:
                p[player]+=1
                y[i][final_row][final_col] =0
                break
        #print(y)
        if np.copy(np.array(y)).tobytes() in story:
            story[np.array(y).tobytes()] +=1
        else:
            story[np.array(y).tobytes()]=1
        return(y,(player+1)%4,story,p)
        
        
                
if __name__ == "__main__":
   
    board = GameBoard()
  
    board.addpiece("wknight1",1, 3,0)
    board.addpiece("wknight2",1, 3,1)
    board.addpiece("wknight3",1, 3,2)
    board.addpiece("wknight4", 1, 3,3)
    board.addpiece("bknight1",-1, 0,0)
    board.addpiece("bknight2",-1, 0,1)
    board.addpiece("bknight3",-1, 0,2)
    board.addpiece("bknight4",-1, 0,3)
    print(board.white)
    print(board.board)
 
    
    
   