
import numpy as np

class Environment:
	grid = []
	dimen = 0
	def __init__(self, array, dimen):
		self.grid = array
		self.dimen = dimen

	def printGrid(self):
		for i in range(0,self.dimen):
			if i != 0 :
				print('\n')
			for j in range(0,self.dimen):
				print(self.grid[i][j], end=" ")
		print('\n')

	def getGridXYVal(self, x, y):
		return self.grid[x][y]
