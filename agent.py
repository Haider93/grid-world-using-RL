import numpy as np
from env import Environment
from memory import Memory
from random import randint
import operator


class Agent:

	def __init__(self,dimen, q_matrix, agent_position, discounted_future_reward = 0.0,
				 reward = 0.0, alpha = 0.5, gamma = 0.8, epsilon = 0.5,
				 gamma_decay_rate = 0.3,choose_action = "right", episode_state = False):
		self.dimen = dimen
		self.q_matrix = q_matrix
		self.reward = reward
		self.discounted_future_reward = discounted_future_reward
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.gamma_decay_rate = gamma_decay_rate
		self.actions_allowed = [0, 1, 2, 3] #left : 0, right : 1, up : 2, down : 3
		self.choose_action = choose_action
		self.agent_position = agent_position
		self.episode_state = episode_state

	def printQMatrix(self):
		print(np.matrix(self.q_matrix))
		# for i in range(0, r):
		# 	for j in range(0, c):
		# 		print(self.q_matrix[i][j])
		# 		if j == c-1:
		# 			print('\n')

	def QValue(self, a):
		return self.q_matrix[(self.dimen * self.agent_position[0]) + self.agent_position[1]][a]

	def update_Q_Matrix(self, a, value):
		'''general formula for finding equivalent index value for 2d array cell position'''
		'''index(one-d-array) = dimension of two-d-array * i + j'''
		'''two-d-array is square matrix'''
		'''row value and column value starts with 0 and ends with two-d-array-dimension - 1'''
		self.q_matrix[(self.dimen * self.agent_position[0]) + self.agent_position[1]][a] += value

	def pick_action(self):
		row_pos = self.agent_position[0]
		col_pos = self.agent_position[1]
		print("dimension of env grid ", self.dimen)
		print("Row : ",row_pos)
		print("Col : ",col_pos)

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
		if(corner_case == False):
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


	def MaxQValueGeneratingAction(self):
		dictionary = {}
		for i in range(0, len(self.actions_allowed)):
			dictionary[self.actions_allowed[i]] = self.QValue(self.actions_allowed[i])

		#sorted(dictionary.items(), key=lambda x: x[1])
		return max(dictionary, key=dictionary.get)
		#key = dictionary.keys()[-1]
		#return key

	def executeAction(self):
		'''Execute actions available according epsilon greedy policy'''
		'''update q table after executing each action'''
		currentX = self.agent_position[0]
		currentY = self.agent_position[1]
		print("Agent is inside : ", self.agent_position)
		self.actions_allowed = Agent.pick_action(self)

		'''balance between exploration and exploitation'''
		'''if exploiting use next highest q value generating action out of allowed actions'''
		'''if exploring use random action out of actions allowed'''
		if randint(0, 1) < self.epsilon: #exploiting
			act = self.MaxQValueGeneratingAction()
		else: 		#exploring
			act = self.actions_allowed[randint(0, len(self.actions_allowed) - 1)]

		print("Action Taken : ", act)

		if act == 0:
			currentY -= 1
			print("Agent has now taken left action")
		elif act == 1:
			currentY += 1
			print("Agent has now taken right action")
		elif act == 2:
			currentX -= 1
			print("Agent has now taken up action")
		elif act == 3:
			currentX += 1
			print("Agent has now taken down action")

		if env1.getGridXYVal(currentX, currentY) == '|G|':
			'''end of episode'''
			self.reward += 100
			print("Agent reached inside terminal state : ", self.agent_position)
			'''here input code for estimating total future reward could be obtained from this new state 
			using bellman eq'''
			self.update_Q_Matrix(act, self.reward)
			self.agent_position[0] = currentX
			self.agent_position[1] = currentY
			self.episode_state = True
			return self.reward

		elif env1.getGridXYVal(currentX, currentY) == '|_|':
			#if mem.isStateVisited([currentX, currentY]) == False:
			self.agent_position[0] = currentX
			self.agent_position[1] = currentY
			print("Agent reached inside  : ", self.agent_position)
			mem.setStateVisited(self.agent_position, act)
			self.reward += -1 + self.gamma * self.executeAction()  # recursion here
			#return (self.gamma * self.reward)

		elif env1.getGridXYVal(currentX, currentY) == '|-|':
			'''Need episode termination to make the agent learn not to bump wall'''
			print("Agent recently bumped to the wall")
			mem.setStateVisited(self.agent_position, act)
			self.reward += -100 + self.gamma * self.executeAction()  # recursion here
			#self.reward -= 100
			#return (self.gamma * self.reward)

		return (self.gamma * self.reward)


array1 = [['|S|','|_|','|_|'],['|_|','|-|','|_|'],['|_|','|_|','|G|']]
#print("Length of enviroment array ",len(array1))
env1 = Environment(array1, 3)
#env1.printGrid()

mem = Memory()

q_matrix = np.zeros([((len(array1) * len(array1))), 4], dtype=float)
agent = Agent(len(array1), q_matrix, [0, 0])
#agent.pick_action()

n_train_episodes = 0
reward = 0

while n_train_episodes < 1000:
	while True:
		print("Training on episode : ", n_train_episodes)
		rl_agent = Agent(len(array1), agent.q_matrix, [0, 0])
		rl_agent.episode_state = False
		reward = rl_agent.executeAction()
		if rl_agent.episode_state == True:
			print("Episode : "+str(n_train_episodes)+" Terminated")
			print("reward attained : " + str(reward))
			print("Q matrix will look like :"+'\n')
			agent.printQMatrix()
			break
	n_train_episodes += 1


print("Total Reward achieved by agent : ",reward)
agent.printQMatrix()



