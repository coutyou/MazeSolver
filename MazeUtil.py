import os, sys, time, datetime, json, random
import numpy as np
from queue import Queue
import matplotlib.pyplot as plt
import matplotlib as mpl

visited_mark = 4  
rat_mark = 0.5   
MAZE_SIZE = 6
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

# Actions dictionary
actions_dict = {
    LEFT: 'left',
    UP: 'up',
    RIGHT: 'right',
    DOWN: 'down',
}

num_actions = len(actions_dict)

# Exploration factor
epsilon = 0.1

def get_neighbor(node, maze):
    size = maze.shape[0]
    r = node[0]
    c = node[1]
    res = []
    if r+1 != size and maze[r+1, c] == 1.:
        res.append((r+1, c))
    if r-1 != -1 and maze[r-1, c] == 1.:
        res.append((r-1, c))
    if c+1 != size and maze[r, c+1] == 1.:
        res.append((r, c+1))
    if c-1 != -1 and maze[r, c-1] == 1.:
        res.append((r, c-1))
    return res

def check_maze(maze,start=(0,0)):
    openList = Queue()
    closeList = []
    openList.put(start)
    closeList.append(start)
    while not openList.empty():
        cur = openList.get()               # 弹出元素
        if cur == (maze.shape[0]-1, maze.shape[1]-1):
            return True
        for neighbor_node in get_neighbor(cur, maze):          # 遍历元素的邻接节点
            if neighbor_node not in closeList:     # 若邻接节点没有入过队，加入队列并登记
                closeList.append(neighbor_node)
                openList.put(neighbor_node)
    return False

def generate_maze(size=6):
    maze = np.array([[random.randint(0,1) for j in range(size)] for i in range(size)])
    wall_index = list(zip(np.where(maze == 1.)[0],np.where(maze == 1.)[1]))
    if (0,0) in wall_index:
        wall_index.remove((0,0))
    if (size-1,size-1) in wall_index:
        wall_index.remove((size-1,size-1))
    trap_index = np.random.choice(range(len(wall_index)), int(0.2*len(wall_index)), replace=False)
    for index in trap_index:
        r,c = wall_index[index]
        maze[r][c] = -1
    maze[0,0] = 1
    maze[-1,-1] = 1
    while not check_maze(maze):
        maze = np.array([[random.randint(0, 1) for j in range(size)] for i in range(size)])
        wall_index = list(zip(np.where(maze == 1.)[0],np.where(maze == 1.)[1]))
        if (0,0) in wall_index:
            wall_index.remove((0,0))
        if (size-1,size-1) in wall_index:
            wall_index.remove((size-1,size-1))
        trap_index = np.random.choice(range(len(wall_index)), int(0.2*len(wall_index)), replace=False)
        for index in trap_index:
            r,c = wall_index[index]
            maze[r][c] = -1
        maze[0,0] = 1
        maze[-1,-1] = 1
    return maze.astype(np.float64)

# maze is a 2d Numpy array of floats between 0.0 to 1.0
# 1.0 corresponds to a free cell, and 0.0 an occupied cell
# rat = (row, col) initial rat position (defaults to (0,0))

class Qmaze(object):
    def __init__(self, maze, rat=(0,0)):
        self._maze = maze
        nrows, ncols = self._maze.shape
        self.target = (nrows-1, ncols-1)   # target cell where the "cheese" is
        self.free_cells = [(r,c) for r in range(nrows) for c in range(ncols) if self._maze[r,c] == 1.]
        self.free_cells.remove(self.target)
        to_del = []
        for cell in self.free_cells:
            if not check_maze(self._maze, cell):
                to_del.append(cell)
        for cell in to_del:
            self.free_cells.remove(cell)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if not rat in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        nrows, ncols = self.maze.shape
        row, col = rat
        self.maze[row, col] = rat_mark
        self.state = (row, col, 'start')
        self.min_reward = -0.05 * self.maze.size
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == LEFT:
                ncol -= 1
            elif action == UP:
                nrow -= 1
            if action == RIGHT:
                ncol += 1
            elif action == DOWN:
                nrow += 1
        else:                  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 1.0
        if self.maze[rat_row][rat_col] == -1:
            return self.min_reward - 1
        if mode == 'blocked':
            return self.min_reward - 1
        if (rat_row, rat_col) in self.visited:
            return -0.3
        if mode == 'invalid':
            return -0.75
        if mode == 'valid':
            return -0.05

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r,c] > 0.0:
                    canvas[r,c] = 1.0
        # draw the rat
        row, col, _ = self.state
        canvas[row, col] = rat_mark
        return canvas

    def game_status(self):
        if self.total_reward < self.min_reward:
            return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows-1 and rat_col == ncols-1:
            return 'win'

        return 'not_over'

    def valid_actions(self, cell=None):
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(1)
        elif row == nrows-1:
            actions.remove(3)

        if col == 0:
            actions.remove(0)
        elif col == ncols-1:
            actions.remove(2)

        # wall
        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(1)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(3)   

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(0)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(2)

        return actions

def show(qmaze, ax=None, updateCanvas=None):
    nrows, ncols = qmaze.maze.shape
    if not ax:
        ax = plt.gca()
        plt.grid('on')
    else:
        ax.grid('on')
    ax.set_xticks(np.arange(0.5, nrows, 1))
    ax.set_yticks(np.arange(0.5, ncols, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(qmaze.maze)
    has_visited = False
    for row,col in qmaze.visited:
        canvas[row,col] = visited_mark
        has_visited = True
    rat_row, rat_col, _ = qmaze.state
    canvas[rat_row, rat_col] = 3   # rat cell
    canvas[nrows-1, ncols-1] = 2 # cheese cell
    
    if not has_visited:
        colors = ['red','black','white','yellow','blue']
    else:
        colors = ['red','black','white','yellow','blue','gray']
    cmap = mpl.colors.ListedColormap(colors)
    
    if not ax:
        img = plt.imshow(canvas, interpolation='none', cmap=cmap)
    else:
        img = ax.imshow(canvas, interpolation='none', cmap=cmap)
        updateCanvas()
    return img

def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        if not action:
            return False

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if np.all(envstate == prev_envstate):
            return False
        elif game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

def show_game(model, qmaze):
    qmaze.reset(qmaze.free_cells[0])
    envstate = qmaze.observe()
    while True:
        plt.clf()
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])

        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        show(qmaze)
        plt.pause(0.5)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False

def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not play_game(model, qmaze, cell):
            return False
    return True