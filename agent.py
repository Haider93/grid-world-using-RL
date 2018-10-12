import numpy as np
from env import Environment
from random import randint

reward = 0

class Agent:

	def __init__(self,dimen, q_matrix, agent_position, reward = 0, alpha = 0.5, gamma = 0.8, prob_random_action = 0.5,
				 gamma_decay_rate = 0.3,choose_action = "right"):
		self.dimen = dimen
		self.q_matrix = q_matrix
		self.reward = reward
		self.alpha = alpha
		self.gamma = gamma
		self.prob_random_action = prob_random_action
		self.gamma_decay_rate = gamma_decay_rate
		self.actions_allowed = ["left","right","up","down"]
		self.choose_action = choose_action
		self.agent_position = agent_position

	def printQMatrix(self, matrix, r, c):
		for i in range(0,r):
			if i != 0 :
				print('\n')
			for j in range(0,c):
				print(matrix[i][j])

	def QFunction(self, s, a):
		return self.q_matrix[s][a]

	def update_Q_Matrix(self,a,value):
		'''general formula for finding equivalent index value for 2d array cell position'''
		'''index(one-d-array) = dimension of two-d-array * i + j'''
		'''two-d-array is square matrix'''
		'''row value and column value starts with 0 and ends with two-d-array-dimension - 1'''
		self.q_matrix[(self.dimen * self.agent_position[0]) + self.agent_position[1]][a] = value


	def reward_function(self, s, a, next_s):
		'''total_reward at s = current reward by action a + gamma times q value at next state'''
		reward = self.q_matrix[s][a] + self.gamma * self.q_matrix[next_s][a]
		return reward



	def Qlearning(self, s, a, r, next_s):
		'''Implementing epsilon greedy algotrithm'''

		pass

	def pick_action(self):
		row_pos = self.agent_position[0]
		col_pos = self.agent_position[1]
		print("dimension of env grid ", self.dimen)
		print("Row : ",row_pos)
		print("Col : ",col_pos)

		corner_case = False
		'''corner cases of grid'''
		if row_pos == 0 and col_pos == 0:
			self.actions_allowed = ["right", "down"]
			corner_case = True
		elif row_pos == 0 and col_pos == self.dimen - 1:
			self.actions_allowed = ["left", "down"]
			corner_case = True
		elif row_pos == self.dimen - 1 and col_pos == 0:
			self.actions_allowed = ["right","up"]
			corner_case = True
		elif row_pos == self.dimen - 1 and col_pos == self.dimen - 1:
			self.actions_allowed = ["left","up"]
			corner_case = True

		'''edge cases'''
		if(corner_case == False):
			for i in range(0, self.dimen - 1):
				'''first row and all columns except first and last column'''
				if ((row_pos == 0 and col_pos == i and col_pos != 0) or
						(row_pos == 0 and col_pos == i and col_pos != self.dimen - 1)):
					self.actions_allowed = ["left","right","down"]
				elif ((row_pos == self.dimen - 1 and col_pos == i and col_pos != 0) or
					  (row_pos == self.dimen - 1 and col_pos == i and col_pos != self.dimen - 1)):
					self.actions_allowed = ["left","right","up"]
				elif ((row_pos == i and row_pos != 0 and col_pos == 0) or
					  (row_pos == i and row_pos != self.dimen - 1 and col_pos == 0)):
					self.actions_allowed = ["right","up","down"]
				elif ((row_pos == i and row_pos != 0 and col_pos == self.dimen - 1) or
					  (row_pos == i and row_pos != self.dimen - 1 and col_pos == self.dimen - 1)):
					self.actions_allowed = ["left","up","down"]
		print("Actions allowed to agent are : ", self.actions_allowed)
		return self.actions_allowed

	def executeAction(self):
		'''Execute actions available according epsilon greedy policy'''
		'''update q table after executing each action'''
		currentX = self.agent_position[0]
		currentY = self.agent_position[1]
		self.actions_allowed = Agent.pick_action(self)
		act = self.actions_allowed[randint(0, len(self.actions_allowed))]
		if act == "left":
			currentX -= 1
		elif act == "right":
			currentX += 1
		elif act == "up":
			currentY -= 1
		elif act == "down":
			currentY += 1

		self.agent_position[0] = currentX
		self.agent_position[1] = currentY

		if env1.getGridXYVal(currentX, currentY) != '|G|':
			'''end of episode'''
			self.reward += 100
			self.update_Q_Matrix(act, self.reward)
			return self.reward
		elif env1.getGridXYVal(currentX, currentY) != '|_|':
			self.reward -= 1
		elif env1.getGridXYVal(currentX, currentY) != '|-|':
			self.reward -= 100








array1 = [['|S|','|_|','|_|'],['|_|','|-|','|_|'],['|_|','|_|','|G|']]
env1 = Environment(array1,3)
env1.printGrid()

q_matrix = np.zeros((3,3), dtype=int)
agent = Agent(4,q_matrix, [0,0])
agent.pick_action()

while True:
	reward = agent.executeAction()
	if reward != 0:
		break

print("Total Reward achieved by agent : ",reward)


