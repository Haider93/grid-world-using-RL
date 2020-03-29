import numpy as np
from env import Environment
from memory import Memory
from random import randint
import time
import BayesianOptimization
from numpy.linalg import inv
from numpy import linalg as LA
import operator
import matplotlib.pyplot as plt
from matplotlib import gridspec

class Agent:

    def __init__(self, dimen, q_matrix, agent_position, env, eligibility=[],
                 discounted_future_reward=0.0, reward=0.0, alpha=0.5, gamma=0.3, lamb=0.95, epsilon=0.5,
                 gamma_decay_rate=0.3, choose_action="right", episode_state=False):
        self.dimen = dimen
        self.q_matrix = q_matrix
        self.reward = reward
        self.discounted_future_reward = discounted_future_reward
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.epsilon = epsilon
        self.gamma_decay_rate = gamma_decay_rate
        self.actions_allowed = [0, 1, 2, 3]  # left : 0, right : 1, up : 2, down : 3
        self.choose_action = choose_action
        self.agent_position = agent_position
        self.env = env
        self.eligibility = eligibility
        self.agent_prev_position = [0, 0]
        self.episode_state = episode_state
        self.counter = 0

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

    def update_Q_Matrix_xy(self, a, reward, prev_position, curr_position):
        self.eligibility *= self.lamb * self.gamma
        self.eligibility[(self.dimen * prev_position[0]) + prev_position[1]] += 1
        update = reward + self.gamma * (self.q_matrix[(self.dimen * curr_position[0])+curr_position[1]])[self.MaxQValueGeneratingAction(curr_position[0], curr_position[1])] - self.q_matrix[(self.dimen * prev_position[0]) + prev_position[1]][a]
        update = self.alpha * update
        self.q_matrix[(self.dimen * prev_position[0]) + prev_position[1]][a] = update * self.eligibility[(self.dimen * prev_position[0]) + prev_position[1]]



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
        if self.env.getGridXYVal(x, y) == '|_|' and mem.isStateVisited([x, y]) == False:
            agent_pos[0] = x
            agent_pos[1] = y
            #mem.setStateVisited(x,y)
            self.discounted_future_reward += 0 + self.gamma * self.future_discounted_reward(agent_pos)
        elif self.env.getGridXYVal(x, y) == '|-|' and mem.isStateVisited([x, y]) == False:
            self.discounted_future_reward += -1
        else:
            self.discounted_future_reward += -1
            agent_pos[0] = x
            agent_pos[1] = y
        self.counter += 1
        return self.discounted_future_reward

    def QLearn(self):
        x = self.agent_position[0]
        y = self.agent_position[1]
        temp_pos = [0,0]
        while True:
            # x = self.agent_position[0]
            # y = self.agent_position[1]
            self.agent_prev_position = self.agent_position
            self.actions_allowed = Agent.pick_action(self, self.agent_position[0], self.agent_position[1])
            if randint(0, 1) < self.epsilon:  # exploiting
                act = self.MaxQValueGeneratingAction(self.agent_position[0], self.agent_position[1])
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

            if x == self.dimen-1 and y == self.dimen-1:
                self.reward = self.reward + 10
                self.update_Q_Matrix_xy(act, self.reward, self.agent_prev_position, [x, y])
                rl_agent.episode_state = True
                return self.reward, self.q_matrix

            elif self.env.getGridXYVal(x, y) == '|_|':
                temp_pos = [x,y]
                #self.reward += 1 + self.future_discounted_reward(temp_pos)
                self.reward = self.reward + 1
                #q_pos_x, q_pos_y = self.find_from_state(act, self.agent_position[0], self.agent_position[1])
                self.update_Q_Matrix_xy(act, self.reward, self.agent_prev_position, [x, y])
                self.agent_position[0] = x
                self.agent_position[1] = y
                #mem.setStateVisited(x, y)
                #self.agent_prev_position = self.agent_position
            elif self.env.getGridXYVal(x, y) == '|-|':
                self.reward = self.reward - 10
                self.update_Q_Matrix_xy(act, self.reward, self.agent_prev_position, self.agent_position)
                x = self.agent_position[0]
                y = self.agent_position[1]
                #mem.setStateVisited(x, y)
            else:
                self.reward = self.reward - 1
                self.update_Q_Matrix_xy(act, self.reward, self.agent_prev_position, [x, y])
                self.agent_position[0] = x
                self.agent_position[1] = y

    def index(self,index,j):
        self.agent_position = [index, j]
        self.actions_allowed = Agent.pick_action(self, index, j)
        max = self.q_matrix[index][self.actions_allowed[0]]
        ind = self.actions_allowed[0]
        for i in range(1, len(self.actions_allowed)):
            if max < self.q_matrix[index][self.actions_allowed[i]]:
                ind = self.actions_allowed[i]
                max = self.q_matrix[index][self.actions_allowed[i]]
        return ind

    def test(self,agent_position):
        self.agent_position = agent_position
        policy = ["Policy-->"]
        while True:
            if self.agent_position[0] == self.dimen-1 and self.agent_position[1] == self.dimen-1:
                print("AT Goal Tested")
                return policy
            if self.index(self.agent_position[0],self.agent_position[1]) == 0:
                policy.append("Left-->")
                if self.agent_position[0] >= 0 and self.agent_position[0] < self.dimen:
                    self.agent_position[0] -= 1
            elif self.index(self.agent_position[0],self.agent_position[1]) == 1:
                policy.append("Right-->")
                if self.agent_position[0] >= 0 and self.agent_position[0] < self.dimen:
                    self.agent_position[0] += 1
            elif self.index(self.agent_position[0],self.agent_position[1]) == 2:
                policy.append("Up-->")
                if self.agent_position[1] >= 0 and self.agent_position[1] < self.dimen:
                    self.agent_position[1] -= 1
            elif self.index(self.agent_position[0],self.agent_position[1]) == 3:
                policy.append("Down-->")
                if self.agent_position[1] >= 0 and self.agent_position[1] < self.dimen:
                    self.agent_position[1] += 1
        return policy


    ##Bayesian optimization
    def black_box_func(self, x, y):
        return 0.5*x + 0.5*y + 1.0



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

pbounds = {'x': (0, len(grid1)), 'y': (0, len(grid1))}




## function approximation instead of q table
file = open("grid_world_output.txt", "a")

while True:
    mem = Memory()
    q_matrix = np.zeros([((len(grid1) * len(grid1))), 4], dtype=float)
    eligibility = np.zeros(len(grid1) * len(grid1))
    # 4 is number of actions L,R,U,D
    choice = int(input("Select option"+"\n1. Q Learning"+"\n2. Function Approximation"+"\n3. Quit"+"\n"))
    print(choice)
    n_train_episodes = 0
    n_eval_episodes = 0
    reward = 0


    if choice == 1:
        file.write("Q learning method \n\n")
        rl_agent = Agent(len(grid1), q_matrix, [0, 0], env1, eligibility)
        while n_train_episodes < 1:

            rl_agent.episode_state = False
            ##Q learn
            reward, q_matrix = rl_agent.QLearn()
            if rl_agent.episode_state == True:
                print("Episode : " + str(n_train_episodes) + " Terminated")
                print("reward attained : " + str(reward))
                print("Q matrix will look like :" + '\n')
                #agent.printQMatrix()
                print(np.matrix(q_matrix))
                #mem.clearMemory()
            n_train_episodes += 1
        file.write(str(np.matrix(q_matrix))+str("\n"))
        #testing phase
        # while n_eval_episodes < 1:
        #     print("Testing on episode : ", n_eval_episodes)
        #     policy = rl_agent.test([0, 0])
        #     print("Policy followed-->", policy)
        #     n_eval_episodes += 1
        # file.write(str(policy) + str("\n"))
        break
    elif choice == 2:
        file.write("FA method \n\n")
        while n_train_episodes < 5:
            rl_agent = Agent(len(grid1), q_matrix, [0, 0], env1, eligibility)
            rl_agent.episode_state = False
            ##FA using BO starts
            optimizer = BayesianOptimization(
                f=rl_agent.black_box_func,
                pbounds=pbounds,
                verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=1,
            )
            optimizer.maximize(
                init_points=2,
                n_iter=3,
            )
            print("", optimizer.max)
            ##BO ends
            if rl_agent.episode_state == True:
                print("Episode : " + str(n_train_episodes) + " Terminated")
                print("reward attained : " + str(reward))
            n_train_episodes += 1
        #file.write(str(weight_vector)+str("\n"))
        break
    elif choice == 3:
        print("Exiting..")
        time.sleep(3)
        break
    else:
        print("Choice not valid")

    file.close()



# print("Total Reward achieved by agent during training : ",reward)
# agent.printQMatrix()
