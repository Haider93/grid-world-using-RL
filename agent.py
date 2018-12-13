import numpy as np
from env import Environment
from memory import Memory
from random import randint
from numpy.linalg import inv
from numpy import linalg as LA
import operator
weight_vector_dimen = 4  # one for each feature vector
weight_vector = np.zeros(weight_vector_dimen, dtype=float)

class Agent:

    def __init__(self, dimen, q_matrix, agent_position, env, action_vector=[], weight_vector=[0.0, 0.0, 0.0, 0.0],
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
        self.env = env
        self.agent_prev_position = [0, 0]
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
        # Q Update function
        # if a not in self.actions_allowed:
        #     return
        self.q_matrix[(self.dimen * x) + y][a] = value \
                                                 + self.alpha \
                                                 * self.q_matrix[(self.dimen * x) + y][self.MaxQValueGeneratingAction(x, y)] \
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
            self.discounted_future_reward += 10
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
        if self.env.getGridXYVal(x, y) == '|_|' and mem.isStateVisited(x, y) == False:
            agent_pos[0] = x
            agent_pos[1] = y
            #mem.setStateVisited(x,y)
            self.discounted_future_reward += 0 + self.gamma * self.future_discounted_reward(agent_pos)
        elif self.env.getGridXYVal(x, y) == '|-|' and mem.isStateVisited(x, y) == False:
            self.discounted_future_reward += -1
        else:
            self.discounted_future_reward += 0
            agent_pos[0] = x
            agent_pos[1] = y
        return self.discounted_future_reward

    def QLearn(self):
        x = self.agent_position[0]
        y = self.agent_position[1]
        temp_pos = [0,0]
        while True:
            if x == self.dimen-1 and y == self.dimen-1:
                self.reward += 10
                self.update_Q_Matrix_xy(act, self.reward, x, y)
                rl_agent.episode_state = True
                return self.reward, self.q_matrix

            x = self.agent_position[0]
            y = self.agent_position[1]
            self.agent_prev_position = self.agent_position
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

            if self.env.getGridXYVal(x, y) == '|_|':
                temp_pos = [x,y]
                self.reward += 0 + self.future_discounted_reward(temp_pos)
                #q_pos_x, q_pos_y = self.find_from_state(act, self.agent_position[0], self.agent_position[1])
                self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])
                self.agent_position[0] = x
                self.agent_position[1] = y
                #mem.setStateVisited(x, y)
                #self.agent_prev_position = self.agent_position
            elif self.env.getGridXYVal(x, y) == '|-|':
                self.reward += -1
                self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])
                #mem.setStateVisited(x, y)
            else:
                self.reward += 0
                self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])
                self.agent_position[0] = x
                self.agent_position[1] = y




    def find_from_state(self, action, xx, yy):
        if action == 0:
            yy += 1
        elif action == 1:
            yy -= 1
        elif action == 2:
            xx += 1
        elif action == 3:
            xx -= 1
        if xx > self.dimen - 1:
            xx -= 1
        elif yy > self.dimen -1:
            yy -= 1
        elif xx < 0:
            xx += 1
        elif yy < 0:
            yy += 1
        return xx,yy
    # def executeAction(self):
    #     '''Execute actions available according epsilon greedy policy'''
    #     '''update q table after executing each action'''
    #     '''balance between exploration and exploitation'''
    #     '''if exploiting use next highest q value generating action out of allowed actions'''
    #     '''if exploring use random action out of actions allowed'''
    #     currentX = self.agent_position[0]
    #     currentY = self.agent_position[1]
    #     self.agent_prev_position = self.agent_position
    #     self.actions_allowed = Agent.pick_action(self)
    #     if randint(0, 1) < self.epsilon:  # exploiting
    #         act = self.MaxQValueGeneratingAction()
    #     else:  # exploring
    #         act = self.actions_allowed[randint(0, len(self.actions_allowed) - 1)]
    #
    #     if act == 0:
    #         currentY -= 1
    #         print("Agent has now taken left action")
    #         self.action_vector = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     elif act == 1:
    #         currentY += 1
    #         print("Agent has now taken right action")
    #         self.action_vector = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     elif act == 2:
    #         currentX -= 1
    #         print("Agent has now taken up action")
    #         self.action_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    #     elif act == 3:
    #         currentX += 1
    #         print("Agent has now taken down action")
    #         self.action_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    #
    #     if mem.isStateVisited(currentX, currentY) == True and self.agent_position != self.agent_prev_position:
    #         print("Agent revisited", self.agent_position)
    #         return self.reward
    #
    #     if currentX == 0 and currentY == 2:
    #         x = 0
    #
    #     if env1.getGridXYVal(currentX, currentY) == '|G|':
    #         '''end of episode'''
    #         self.reward += 10
    #         '''here input code for estimating total future reward could be obtained from this new state using bellman eq'''
    #         # self.update_Q_Matrix(act, self.reward)
    #         mem.setStateVisited(currentY, currentY)
    #         self.update_Q_Matrix_xy(act, self.reward, self.agent_position[0], self.agent_position[1])
    #         self.modifyActionVector('|G|')
    #         self.weight_vector = np.matmul((np.array(self.action_vector) / LA.norm(np.array(self.action_vector))), (
    #                 self.reward + self.gamma * np.array(np.outer(self.action_vector, self.weight_vector))))
    #         # FA ends
    #         self.agent_position[0] = currentX
    #         self.agent_position[1] = currentY
    #         print("Agent reached inside terminal state : ", self.agent_position)
    #         self.episode_state = True
    #         weight_vector = self.weight_vector
    #         print("Weight vector is : ", self.weight_vector)
    #         return self.reward
    #
    #     elif env1.getGridXYVal(currentX, currentY) == '|_|':
    #         self.agent_position[0] = currentX
    #         self.agent_position[1] = currentY
    #         mem.setStateVisited(currentY, currentY)
    #
    #
    #         self.reward += 1 + self.gamma * self.executeAction()  # recursion here
    #         self.update_Q_Matrix_xy(act, self.reward, self.agent_prev_position[0], self.agent_prev_position[1])
    #
    #         print("Agent reached inside  : ", self.agent_position)
    #         # calculate value of product of feature and weight vector
    #         # FA (function approximation) using bellman equation
    #         self.modifyActionVector('|_|')
    #         self.weight_vector = np.matmul((np.array(self.action_vector) / LA.norm(np.array(self.action_vector))), (
    #                 self.reward + self.gamma * np.array(np.outer(self.action_vector, self.weight_vector))))
    #     # FA ends
    #     # return (self.gamma * self.reward)
    #
    #     elif env1.getGridXYVal(currentX, currentY) == '|-|':
    #         '''Need episode termination to make the agent learn not to bump wall'''
    #         print("Agent recently bumped to the wall")
    #
    #         self.reward += -1 + self.gamma * self.executeAction()  # recursion here
    #         self.update_Q_Matrix_xy(act, self.reward)
    #
    #         # calculate value of product of feature and weight vector
    #         # FA (function approximation) using bellman equation
    #         self.modifyActionVector('|-|')
    #         self.weight_vector = np.matmul((np.array(self.action_vector) / LA.norm(np.array(self.action_vector))), (
    #                 self.reward + self.gamma * np.array(np.outer(self.action_vector, self.weight_vector))))
    #     # FA ends
    #     # self.reward -= 100
    #     # return (self.gamma * self.reward)
    #     return (self.gamma * self.reward)


##test
grid1 = [['|S|', '|_|', '|_|'], ['|_|', '|-|', '|_|'], ['|_|', '|_|', '|G|']]
grid2 = [['|S|', '|_|', '|_|', '|_|'], ['|_|', '|-|', '|_|', '|_|'], ['|_|', '|_|', '|_|', '|_|'],['|_|', '|_|', '|_|', '|G|']]
# print("Length of enviroment array ",len(array1))
env1 = Environment(grid1, 3)
env2 = Environment(grid2, 4)
# env1.printGrid()

#q_matrix = np.zeros([((len(array1) * len(array1))), 4], dtype=float)
#agent = Agent(len(array1), q_matrix, [0, 0])
# agent.pick_action()

n_train_episodes = 0
n_eval_episodes = 0
reward = 0

feature_vector_dimen = 12  # actions * features
action_vector_dimen = 4  # left, right, up, down

feature_vector = np.zeros(feature_vector_dimen, dtype=float)
# print("vector ", feature_vector_left)
# print("vector ", feature_vector_right)
# print("vector ", feature_vector_up)
# print("vector ", feature_vector_down)

while n_train_episodes < 1000:
    mem = Memory()
    q_matrix = np.zeros([((len(grid2) * len(grid2))), 4], dtype=float)
    # 4 is number of actions L,R,U,D
    while True:
        #print("Training on episode : ", n_train_episodes)
        rl_agent = Agent(len(grid2), q_matrix, [0, 0], env2, feature_vector.transpose())
        rl_agent.episode_state = False
        reward, q_matrix = rl_agent.QLearn()
        if rl_agent.episode_state == True:
            print("Episode : " + str(n_train_episodes) + " Terminated")
            print("reward attained : " + str(reward))
            print("Q matrix will look like :" + '\n')
            #agent.printQMatrix()
            print("Left" + ", Right" + ", Up" + ", Down" + "\n")
            print(np.matrix(q_matrix))
            #mem.clearMemory()
            break
    n_train_episodes += 1

# while n_eval_episodes < 50:
#     while True:
#         # print("Testing on episode : ", n_eval_episodes)
#         break;
#     n_eval_episodes += 1

# print("Total Reward achieved by agent during training : ",reward)
# agent.printQMatrix()
