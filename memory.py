class Memory:
    Dictionary = {}

    def __init__(self):
        pass

    def setStateVisited(self, state_visited, action_taken):
        self.state_visited = state_visited
        self.action_taken = action_taken


    def isStateVisited(self, state_v):
        #state = ''.join(state_v)
        return state_v in self.Dictionary.values()



#mem = Memory()
#mem.setStateVisited([2,2], "up")
#mem.setStateVisited([2,3], "down")
#if mem.isStateVisited([2, 5]) == True:
    #print("Yes, [2,3] has been visited")
#else:
    #print("Not yet")