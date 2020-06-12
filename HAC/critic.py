import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from utils import nn_layer
from baselines.her.normalizer import Normalizer

class Critic():

    def __init__(self, sess, env, layer_number, FLAGS, hparams, learning_rate=0.001, gamma=0.98, tau=0.05):
        """Critic based on HAC implementation.
        Args:
            sess: tensorflow session
            env: environment object containing the Gym envionment
            batch_size (int): size of the training batches
            layer_number (int): number of the layer this actor belongs to
            FLAGS: flags determining how the alogirthm is run
            hparams: hyperparameters set in run.py
            learning_rate (float): learning rate of the actor
            gamma (float): discount factor
            tau (float): polyak averaging coefficient
        """

        self.sess = sess

        self.dimo = env.state_dim
        self.dimg = env.end_goal_dim
        self.dimu = env.subgoal_dim
        self.norm_eps = 0.01
        self.norm_clip = 5

        with tf.variable_scope('critic' + str(layer_number) + '/o_stats') as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)

        with tf.variable_scope('critic' + str(layer_number) + '/g_stats') as vs:
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        with tf.variable_scope('critic' + str(layer_number) + '/u_stats') as vs:
            self.u_stats = Normalizer(self.dimu, self.norm_eps, self.norm_clip, sess=self.sess)

        # tensorboard logging
        self.o_stats_mean = 0
        self.o_stats_std = 0
        self.g_stats_mean = 0
        self.g_stats_std = 0
        self.u_stats_mean = 0
        self.u_stats_std = 0


        self.critic_name = 'critic_' + str(layer_number)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
       
        self.q_limit = -FLAGS.time_scale

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == hparams["layers"] - 1:
            self.goal_dim = env.end_goal_dim
        else:
            self.goal_dim = env.subgoal_dim

        self.loss_val = 0
        self.state_dim = env.state_dim
        self.state_ph = tf.placeholder(tf.float32, shape=(None, env.state_dim), name='state_ph')
        self.goal_ph = tf.placeholder(tf.float32, shape=(None, self.goal_dim))

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            action_dim = env.action_dim
        else:
            action_dim = env.subgoal_dim

        self.action_ph = tf.placeholder(tf.float32, shape=(None, action_dim), name='action_ph')

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # Create critic network
        self.main = self._create_network(self.state_ph, self.goal_ph, self.action_ph)
        self.weights = [v for v in tf.trainable_variables() if self.critic_name in v.op.name]


        self.target = self._create_network(self.state_ph, self.goal_ph, self.action_ph, name = self.critic_name + '_target')
        self.target_weights = [v for v in tf.trainable_variables() if self.critic_name in v.op.name][len(self.weights):]

        self.update_target_weights = \
        [self.target_weights[i].assign(tf.multiply(self.weights[i], self.tau) +
                                                  tf.multiply(self.target_weights[i], 1. - self.tau))
                    for i in range(len(self.target_weights))]
    
        self.wanted_qs = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss = tf.reduce_mean(tf.square(self.wanted_qs - self.main))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        self.gradient = tf.gradients(self.main, self.action_ph)




    def get_Q_value(self,state, goal, action):
        return self.sess.run(self.main,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]

    def get_target_Q_value(self,state, goal, action):
        return self.sess.run(self.target,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })[0]


    def update(self, old_states, old_actions, rewards, new_states, goals, new_actions, is_terminals):

        wanted_qs = self.sess.run(self.target,
                feed_dict={
                    self.state_ph: new_states,
                    self.goal_ph: goals,
                    self.action_ph: new_actions
                })
        
       
        for i in range(len(wanted_qs)):
            if is_terminals[i]:
                wanted_qs[i] = rewards[i]
            else:
                wanted_qs[i] = rewards[i] + self.gamma * wanted_qs[i][0]

            # Ensure Q target is within bounds [-self.time_limit,0]
            wanted_qs[i] = max(min(wanted_qs[i],0), self.q_limit)
            assert wanted_qs[i] <= 0 and wanted_qs[i] >= self.q_limit, "Q-Value target not within proper bounds"


        self.loss_val, _ = self.sess.run([self.loss, self.train],
                feed_dict={
                    self.state_ph: old_states,
                    self.goal_ph: goals,
                    self.action_ph: old_actions,
                    self.wanted_qs: wanted_qs 
                })

        o = np.asarray(old_states)
        g = np.asarray(goals)
        u = np.asarray(old_actions)

        self.o_stats.update(o)
        self.o_stats.recompute_stats()
        self.o_stats_mean = np.mean(self.sess.run([self.o_stats.mean]))
        self.o_stats_std = np.mean(self.sess.run([self.o_stats.std]))
        self.g_stats.update(g)
        self.g_stats.recompute_stats()
        self.g_stats_mean = np.mean(self.sess.run([self.g_stats.mean]))
        self.g_stats_std = np.mean(self.sess.run([self.g_stats.std]))
        self.u_stats.update(u)
        self.u_stats.recompute_stats()
        self.u_stats_mean = np.mean(self.sess.run([self.u_stats.mean]))
        self.u_stats_std = np.mean(self.sess.run([self.u_stats.std]))


        

    def get_gradients(self, state, goal, action):
        grads = self.sess.run(self.gradient,
                feed_dict={
                    self.state_ph: state,
                    self.goal_ph: goal,
                    self.action_ph: action
                })

        return grads[0]

    # Function creates the graph for the critic function.  The output uses a sigmoid, which bounds the Q-values to between [-Policy Length, 0].
    def _create_network(self, state, goal, action, name=None):

        o = self.o_stats.normalize(state)
        g = self.g_stats.normalize(goal)
        u = self.u_stats.normalize(action)
        input = tf.concat(axis=1, values=[o, g, u])

        if name is None:
            name = self.critic_name        

        with tf.variable_scope(name + '_fc_1'):
            fc1 = nn_layer(input, 64)
        with tf.variable_scope(name + '_fc_2'):
            fc2 = nn_layer(fc1, 64)
        with tf.variable_scope(name + '_fc_3'):
            fc3 = nn_layer(fc2, 64)
        with tf.variable_scope(name + '_fc_4'):
            fc4 = nn_layer(fc3, 1, is_output=True)

            # A q_offset is used to give the critic function an optimistic initialization near 0
            output = tf.sigmoid(fc4 + self.q_offset) * self.q_limit

        return output