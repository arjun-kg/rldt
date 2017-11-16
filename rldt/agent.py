
import sys
from pylab import *
# import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import gym
import numpy as np
import tensorflow.contrib.layers as layers
from gym import wrappers
from sklearn.preprocessing import train_test_split as tts
from envs.env_notgym import env


class Agent(object):
    def __init__(self, input_size=4, hidden_size=2, gamma=0.95,
                 action_size=6, lr=0.1, dir='tmp/trial/'):
        # call the cartpole simulator from OpenAI gym package
        self.env = env(X,y)
        # If you wish to save the simulation video, simply uncomment the line below
        # self.env = wrappers.Monitor(self.env, dir, force=True, video_callable=self.video_callable)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.action_size = action_size
        self.lr = lr
        # save the hyper parameters
        self.params = self.__dict__.copy()

        # inputs to the controller
        self.input_pl = tf.placeholder(tf.float32, [None, input_size])
        self.action_pl = tf.placeholder(tf.int32, [None])
        self.reward_pl = tf.placeholder(tf.float32, [None])

        # Here we use a single layered neural network as controller, which proved to be sufficient enough.
        # More complicated ones can be plugged in as well.
        # hidden_layer = layers.fully_connected(self.input_pl,
        #                                      hidden_size,
        #                                      biases_initializer=None,
        #                                      activation_fn=tf.nn.relu)
        # hidden_layer = layers.fully_connected(hidden_layer,
        #                                       hidden_size,
        #                                       biases_initializer=None,
        #                                       activation_fn=tf.nn.relu)
        self.output = layers.fully_connected(self.input_pl,
                                             action_size,
                                             biases_initializer=None,
                                             activation_fn=tf.nn.softmax)


        # responsible output
        self.one_hot = tf.one_hot(self.action_pl, action_size)
        self.responsible_output = tf.reduce_sum(self.output * self.one_hot, axis=1)

        # loss value of the network
        self.loss = -tf.reduce_mean(tf.log(self.responsible_output) * self.reward_pl)

        # get all network variables
        variables = tf.trainable_variables()
        self.variable_pls = []
        for i, var in enumerate(variables):
            self.variable_pls.append(tf.placeholder(tf.float32))

        # compute the gradient values
        self.gradients = tf.gradients(self.loss, variables)

        # update network variables
        solver = tf.train.AdamOptimizer(learning_rate=self.lr)
        # solver = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.95)
        self.update = solver.apply_gradients(zip(self.variable_pls, variables))

    def next_action(self, sess, feed_dict, greedy=False):
        """Pick an action based on the current state.
        Args:
        - sess: a tensorflow session
        - feed_dict: parameter for sess.run()
        - greedy: boolean, whether to take action greedily
        Return:
            Integer, action to be taken.
        """
        ans = sess.run(self.output, feed_dict=feed_dict)[0]
        if greedy:
            return ans.argmax()
        else:
            return np.random.choice(range(self.action_size), p=ans)

    def show_parameters(self):
        """Helper function to show the hyper parameters."""
        for key, value in self.params.items():
            print(key, '=', value)

def discounted_reward(rewards, gamma):
    """Compute the discounted reward."""
    ans = np.zeros_like(rewards)
    running_sum = 0
    # compute the result backward
    for i in reversed(range(len(rewards))):
        running_sum = running_sum * gamma + rewards[i]
        ans[i] = running_sum
    return ans

def one_trial(agent, sess, grad_buffer, reward_itr, i, render = False):
   
    # reset the environment
    s = agent.env.reset()
    for idx in range(len(grad_buffer)):
        grad_buffer[idx] *= 0
    state_history = []
    reward_history = []
    action_history = []
    current_reward = 0

    while True:

        feed_dict = {agent.input_pl: [s]}
        # update the controller deterministically
        greedy = False
        # get the controller output under a given state
        action = agent.next_action(sess, feed_dict, greedy=greedy)
        # get the next states after taking an action
        snext, r, done = agent.env.step(action)

        current_reward += r
        state_history.append(s)
        reward_history.append(r)
        action_history.append(action)
        s = snext

        if done:

            # record how long it has been balancing when the simulation is done
            reward_itr += [current_reward]

            # get the "long term" rewards by taking decay parameter gamma into consideration
            rewards = discounted_reward(reward_history, agent.gamma)

            # normalizing the reward makes training faster
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)

            # compute network gradients
            feed_dict = {
                agent.reward_pl: rewards,
                agent.action_pl: action_history,
                agent.input_pl: np.array(state_history)
            }
            episode_gradients = sess.run(agent.gradients,feed_dict=feed_dict)
            for idx, grad in enumerate(episode_gradients):
                grad_buffer[idx] += grad

            # apply gradients to the network variables
            feed_dict = dict(zip(agent.variable_pls, grad_buffer))
            sess.run(agent.update, feed_dict=feed_dict)

            # reset the buffer to zero
            for idx in range(len(grad_buffer)):
                grad_buffer[idx] *= 0
            break

    return state_history

def main():
    
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	enc = OHE()
	enc.fit(y)
	y = enc.transform(y).toarray()

	e = env(X,y)

	# X_train, X_test, y_train, y_test = tts(X,y,test_size = 0.33, random_seed = 21)

    obt_itr = 10
    max_epoch = 100
    
    # set up figure for animation
    agent = Agent(hidden_size=24, lr=0.2, gamma=0.95, dir=dir)
    agent.show_parameters()

    # tensorflow initialization for neural network controller
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
    tf.global_variables_initializer().run(session=sess)
    grad_buffer = sess.run(tf.trainable_variables())
    tf.reset_default_graph()


    global reward_itr
    reward_itr = []
    args = [agent, sess, grad_buffer, reward_itr, sess, grad_buffer, agent, obt_itr, render]
    # run the optimization 
    plt.show()

if __name__ == "__main__":
   main()