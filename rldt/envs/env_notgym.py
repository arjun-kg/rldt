import numpy as np 

class env(X,y):

	def __init__(self,X,y): 
		self.state = None
		self.X = X
		self.y = y
		self.counter = 0



	def _step(self, action):

		reward = 0
		done = False
		check = np.argmax(self.label)
		if action == 0:
            val = self.datapoint[0]
        elif action == 1:
            val = self.datapoint[1]
        elif action == 2:
            val = self.datapoint[2]
        elif action == 3:
            if check == 0:
                reward = 1
            else:
                reward = -1
            done = True
        elif action == 4:
            if check == 1:
                reward = 1
            else:
                reward = -1
            done = True
        
        elif action == 5:
            if check == 2:
                reward = 1
            else:
                reward = -1
            done = True

        self.state = val
        return self.state,reward, done

    def _reset(self):
		self.datapoint = X[self.counter]
		self.label = y[self.counter]
		self.counter+=1