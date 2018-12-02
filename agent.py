import numpy as np
from env import Environment
from random import randint
from numpy.linalg import inv
from numpy import linalg as LA
import operator

class Agent:

    def __init__(self, dimen, q_matrix, agent_position, action_vector=[], weight_vector=[0.0, 0.0, 0.0, 0.0],
                 discounted_future_reward=0.0, reward=0.0, alpha=0.5, gamma=0.8, epsilon=0.5,
                 gamma_decay_rate=0.3, choose_action="right", episode_state=False):
        self.dimen = dimen
        self.q_matrix = q_matrix
        self.reward = reward
        self.discounted_future_reward = discounted_future_reward
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.gamma_decay_rate = gamma_decay_rate
        self.actions_allowed = [0, 1, 2, 3]  # left : 0, right : 1, up : 2, down : 3
        self.choose_action = choose_action
        self.agent_position = agent_position
        self.agent_prev_position = [0,0]
        self.episode_state = episode_state
        self.weight_vector = weight_vector
        self.action_vector = action_vector

    def printQMatrix(self):
        print(np.matrix(self.q_matrix))

    def QValue(self, a, x, y):
        return self.q_matrix[(self.dimen * x) + y][a]

    def update_Q_Matrix(self, a, value):
        '''general formula for finding equivalent index value for 2d array cell position'''
        '''index(one-d-array) = dimension of two-d-array * i + j'''
        '''two-d-array is square matrix'''
        '''row value and column value starts with 0 and ends with two-d-array-dimension - 1'''
        self.q_matrix[(self.dimen * self.agent_prev_position[0]) + self.agent_prev_position[1]][a] += value

    def update_Q_Matrix_xy(self, a, value, x=0, y=0):
        if x == 0 and y == 0:
            x = self.agent_prev_position[0]
            y = self.agent_prev_position[1]
        # Q Update function
        self.q_matrix[(self.dimen * x) + y][a] = value \
                                                 + self.alpha * self.MaxQValueGeneratingAction(x, y) \
                                                 - self.q_matrix[(self.dimen * x) + y][a]


    def pick_action(self, row = 0, col = 0):
        if row == 0 and col == 0:
            row_pos = self.agent_position[0]
            col_pos = self.agent_position[1]
        else:
            row_pos = row
            col_pos = col

        corner_case = False
        '''corner cases of grid'''
        if row_pos == 0 and col_pos == 0:
            self.actions_allowed = [1, 3]
            corner_case = True
        elif row_pos == 0 and col_pos == self.dimen - 1:
            self.actions_allowed = [0, 3]
            corner_case = True
        elif row_pos == self.dimen - 1 and col_pos == 0:
            self.actions_allowed = [1, 2]
            corner_case = True
        elif row_pos == self.dimen - 1 and col_pos == self.dimen - 1:
            self.actions_allowed = [0, 2]
            corner_case = True

        '''edge cases'''
        if (corner_case == False):
            for i in range(0, self.dimen - 1):
                '''first row and all columns except first and last column'''
                if ((row_pos == 0 and col_pos == i and col_pos != 0) or
                        (row_pos == 0 and col_pos == i and col_pos != self.dimen - 1)):
                    self.actions_allowed = [0, 1, 3]
                elif ((row_pos == self.dimen - 1 and col_pos == i and col_pos != 0) or
                      (row_pos == self.dimen - 1 and col_pos == i and col_pos != self.dimen - 1)):
                    self.actions_allowed = [0, 1, 2]
                elif ((row_pos == i and row_pos != 0 and col_pos == 0) or
                      (row_pos == i and row_pos != self.dimen - 1 and col_pos == 0)):
                    self.actions_allowed = [1, 2, 3]
                elif ((row_pos == i and row_pos != 0 and col_pos == self.dimen - 1) or
                      (row_pos == i and row_pos != self.dimen - 1 and col_pos == self.dimen - 1)):
                    self.actions_allowed = [0, 2, 3]
        print("Actions allowed to agent are : ", self.actions_allowed)
        return self.actions_allowed

    def MaxQValueGeneratingAction(self, x, y):
        dictionary = {}
        for i in range(0, len(self.actions_allowed)):
            dictionary[self.actions_allowed[i]] = self.QValue(self.actions_allowed[i], x, y)

        # sorted(dictionary.items(), key=lambda x: x[1])
        return max(dictionary, key=dictionary.get)

    # key = dictionary.keys()[-1]
    # return key

    def modifyActionVector(self, state):
        for i in range(0, len(self.action_vector)):
            if self.action_vector[i] == 1.0 and i < len(self.action_vector)-2:
                if state == '|G|':
                    self.action_vector[i] = 0.0
                    i+=1
                    self.action_vector[i] = 0.0
                    i+=1
                    self.action_vector[i] = 1.0
                    break;
                elif state == '|_|':
                    self.action_vector[i] = 0.0
                    i += 1
                    self.action_vector[i] = 1.0
                    i += 1
                    self.action_vector[i] = 0.0
                    break;
                elif state == '|-|':
                    self.action_vector[i] = 1.0
                    i += 1
                    self.action_vector[i] = 0.0
                    i += 1
                    self.action_vector[i] = 0.0
                    break;


    def future_discounted_reward(self, agent_pos):
        x = agent_pos[0]
        y = agent_pos[1]
        if x == self.dimen-1 and y == self.dimen-1:
            return self.discounted_future_reward
        self.actions_allowed = Agent.pick_action(self, x, y)
        if randint(0, 1) < self.epsilon:  # exploiting
            act = self.MaxQValueGeneratingAction(x, y)
        else:  # exploring
            act = self.actions_allowed[randint(0, len(self.actions_allowed) - 1)]

        if act == 0:
            y -= 1
        elif act == 1:
            y += 1
        elif act == 2:
            x -= 1
        elif act == 3:
            x += 1
        if env1.getGridXYVal(x, y) == '|_|' and mem.isStateVisited(x, y) == False:
            agent_pos[0] = x
            agent_pos[1] = y
            mem.setStateVisited(x,y)
            self.discounted_future_reward += 1 + self.gamma * self.future_discounted_reward(agent_pos)
        elif env1.getGridXYVal(x, y) == '|-|' and mem.isStateVisited(x, y) == False:
            self.discounted_future_reward += -1
        else:
            self.discounted_future_reward += 0
        return self.discounted_future_reward

    def QLearn(self):
        x = self.agent_position[0]
        y = self.agent_position[1]
        while True:
            if x == self.dimen-1 and y == self.dimen-1:
                self.reward += 10
                self.update_Q_Matrix_xy(act, self.reward, x, y)
                rl_agent.episode_state = True
                return self.reward
            x = self.agent_position[0]
            y = self.agent_position[1]
            self.actions_allowed = Agent.pick_action(self, x, y)
            if randint(0, 1) < self.epsilon:  # exploiting
                act = self.MaxQValueGeneratingAction(x, y)
            else:  # exploring
                act = self.actions_allowed[randint(0, len(self.actions_allowed) - 1)]

            if act == 0:
                #left
                y -= 1
            elif act == 1:
                #right
                y += 1
            elif act == 2:
                #up
                x -= 1
            elif act == 3:
                #down
                x += 1

            if env1.getGridXYVal(x, y) == '|_|':
                self.agent_position[0] = x
                self.agent_position[1] = y
                self.reward += -1 + self.future_discounted_reward(self.agent_position)
                self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])
            elif env1.getGridXYVal(x, y) == '|-|':
                self.reward += -10
                self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])


##test
array1 = [['|S|', '|_|', '|_|'], ['|_|', '|-|', '|_|'], ['|_|', '|_|', '|G|']]
# print("Length of enviroment array ",len(array1))
env1 = Environment(array1, 3)
# env1.printGrid()

q_matrix = np.zeros([((len(array1) * len(array1))), 4], dtype=float)
agent = Agent(len(array1), q_matrix, [0, 0])
# agent.pick_action()

n_train_episodes = 0
n_eval_episodes = 0
reward = 0

feature_vector_dimen = 12  # actions * features
action_vector_dimen = 4  # left, right, up, down

feature_vector = np.zeros(feature_vector_dimen, dtype=float)

while n_train_episodes < 1000:
    while True:
        rl_agent = Agent(len(array1), agent.q_matrix, [0, 0], feature_vector.transpose())
        rl_agent.episode_state = False
        reward = rl_agent.QLearn()
        if rl_agent.episode_state == True:
            print("Episode : " + str(n_train_episodes) + " Terminated")
            print("reward attained : " + str(reward))
            print("Q matrix will look like :" + '\n')
            agent.printQMatrix()
            break
    n_train_episodes += 1

