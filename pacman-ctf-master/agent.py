import copy
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


tfk = tf.keras
tfkl = tfk.layers
tfm = tf.math

class Network(tfk.Model):
    def __init__(self,filters_1=32,filters_2=64, kernel_size_1=5, kernel_size_2=3,num_actions=5,alpha=1e-3): # do I need **args  
        super(Network,self).__init__()
        self.conv_1 = tfkl.Conv2D(filters=filters_1, kernel_size=kernel_size_1,padding='Same',activation="relu")
        self.pool = tfkl.MaxPool2D(pool_size=(2, 2))
        self.conv_2 = tfkl.Conv2D(filters=filters_2,kernel_size=kernel_size_2,padding='Same',activation="relu")
        self.flatten = tfkl.Flatten()
        self.classifier = tfkl.Dense(num_actions,activation="linear")
        self.optimizer = tfk.optimizers.Adam(learning_rate=alpha)
        self.loss = tfk.losses.MSE
        

    def call(self,inputs):
        x = self.conv_1(inputs)
        x = self.pool(x)
        x = self.conv_2(x)
        x = self.flatten(x)
        actions = self.classifier(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, alpha, state_dim, batch_size,
            buffer_size=30000, eps_final=0.01):
        #initialize parameters
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.buffer_size = buffer_size
        self.buffer_cnt = 0 
        self.epsilon = epsilon
        self.eps_final = eps_final
        self.step = 0

        #initialize buffer
        self.state_buffer = np.zeros((self.buffer_size, state_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int32)
        self.next_state_buffer = np.zeros((self.buffer_size, state_dim), dtype=np.float32)

        #initialize networks
        self.NN = Network()
        self.NN.compile(self.NN.optimizer,self.NN.loss)
        self.target_NN = Network()
        self.target_NN.compile(self.target_NN.optimizer,self.target_NN.loss)

    def add_to_buffer(self, state, action, reward, next_state):
        '''This function does stores the experience into a buffer to use it later for training'''
        ind = self.buffer_cnt % self.buffer_size
        self.state_buffer[ind] = state
        self.next_state_buffer[ind] = next_state
        self.reward_buffer[ind] = reward
        self.action_buffer[ind] = action
        self.buffer_cnt += 1

    def get_action(self, state, possible_actions):
        '''gets the action according to an epsilon greedy policy given the NN'''
        if np.random.random() < self.epsilon:
            action = np.random.choice(possible_actions)

        else:
            actions = self.NN(state)
            action = tfm.argmax(actions,axis=1).numpy()[0] 
            if action not in possible_actions:
                return np.random.choice(possible_actions)

            

        return action

    def learn(self):
        '''When we have enough samples in the buffer, we will learn every fith step'''
        if self.step != 0 or self.buffer_cnt < self.batch_size:
            return
        
        #to make sure we do not use 0 values.
        if self.buffer_cnt < self.buffer_size:
            batch = np.random.choice(self.buffer_cnt, self.batch_size, replace=False)
        else:
            batch = np.random.choice(self.buffer_size, self.batch_size, replace=False)

        self.train_network(batch)
    
    def update_step(self):
        '''keep track of steps when there is need to update'''
        self.step +=1
        self.step = self.step % 5
    
    def train_network(self,batch):
        '''Train the network given a random batch from the buffer and using a target Network'''
        #get data
        rewards = self.reward_buffer[batch]
        states = self.state_buffer[batch]
        actions = np.argmax(self.action_buffer[batch],axis=1)
        next_states = self.next_state_buffer[batch]
        idx = np.arange(len(batch))
        
        # compute current Q values and Bellman
        current_val = self.NN(states)
        next_val = self.target_NN(next_states)
        targets = rewards + self.gamma*tfm.max(next_val,axis=1) # Do I have to use the Bellman error here or is that done in fit ? 
        current_val = current_val.numpy()
        current_val[idx,actions] = targets

        #update Network
        NN.fit(states,current_val)

    def update_epsilon(self):
        '''After every episode decrease epsilon by 5%, (e.g explore less and less)'''
        if self.epsilon <= self.eps_final:
            self.epsilon = self.eps_final
        else:
            self.epsilon = self.epsilon*0.995


    def update_network(self):
        '''update the target network'''
        self.target_NN = copy.deepcopy(self.NN)
        "this actually might work better"
        # for a, b in zip(target_NN.variables, NN.variables):
        #     a.assign(b) 

        
        