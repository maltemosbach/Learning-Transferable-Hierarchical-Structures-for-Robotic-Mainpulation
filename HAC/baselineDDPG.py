import tensorflow as tf
import numpy as np
from baselines.her.util import store_args, nn
from baselines.her.normalizer import Normalizer
from collections import OrderedDict
from tensorflow.contrib.staging import StagingArea
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch, convert_episode_to_batch_major)
from baselines.common.mpi_adam import MpiAdam

import time




# Helper function
def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


class DDPG():

    def __init__(self, sess, env, hparams, batch_size, transitions_buffer, erb,
        layer_number, FLAGS, hidden, layers, T, buffer_type='transitions', Q_lr=0.001, pi_lr=0.001, tau=0.05, 
        gamma=0.98, action_l2=1.0, norm_eps=0.01, norm_clip=5, clip_obs=200):
        """The new DDPG policy used inside the HAC algorithm. Code is a variation of the baselines DDPG implementation 
        made to fit the hierarchical framework.

        Args:
            sess: tensorflow session
            env: environment object containing the Gym envionment
            hparams: hyperparameters set in run.py
            transitions_buffer: storing all experiences trainsition-wise
            erb: ExperienceReplayBuffer generating hindsight experiences during sampling
            layer_number: number of this layyer in the HAC hierarchy
            FLAGS: flags determining how the algorithm is run
            hidden (int): number of perceptrons in each layer
            layers (int): number of layers of the neural networks
            Q_lr (float): learning rate of the critic
            pi_lr (float): learning rate of the actor
            tau (float): polyak averaging coefficient
            gamma (float): discount rate
            action_l2 (float): action penalty coefficient in critic loss
            norm_eps (float): epsilon in normalizer
            norm_clip (int/float): clipping range of the normalizer
            clip_obs (int/float): clipping range for observations
        """

        self.sess = sess
        self.scope = "DDPG_layer_" + str(layer_number)

        self.transitions_buffer = transitions_buffer
        self.erb = erb

        # DDPG parameters
        self.norm_eps = norm_eps
        self.norm_clip = norm_clip
        self.hidden = hidden
        self.layers = layers
        self.batch_size = batch_size
        self.Q_lr = Q_lr
        self.pi_lr = pi_lr
        self.gamma = gamma
        self.polyak = 1-tau
        self.clip_return = 50
        self.action_l2 = action_l2
        self.clip_obs = clip_obs

        self.T = T
        self.buffer_type = buffer_type

        # For logging normalizers:
        self.o_stats_mean = 0
        self.o_stats_std = 0
        self.g_stats_mean = 0
        self.g_stats_std = 0

        
        # Exposed for tensorboard logging
        self.critic_loss = 0
        self.actor_loss = 0

        # Defining the dimension of observations, goals and actions depending on environment and layer_number
        self.input_dims = {"o" : 0, "g" : 0, "u" : 0}

        self.input_dims["o"] = env.state_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if layer_number == hparams["layers"] - 1:
            self.input_dims["g"] = env.end_goal_dim
        else:
            self.input_dims["g"] = env.subgoal_dim

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.u_offset = env.action_offset
            self.max_u = env.action_bounds
        else:
            # Determine symmetric range of subgoal space and offset
            self.u_offset = env.subgoal_bounds_offset
            self.max_u = env.subgoal_bounds_symmetric

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.input_dims["u"] = env.action_dim
        else:
            self.input_dims["u"] = env.subgoal_dim


        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        # Prepare staging area for feeding data to the model.
        stage_shapes = OrderedDict()
        for key in sorted(self.input_dims.keys()):
            if key.startswith('info_'):
                continue
            stage_shapes[key] = (None, *input_shapes[key])
        for key in ['o', 'g']:
            stage_shapes[key + '_2'] = stage_shapes[key]
        stage_shapes['r'] = (None,)
        self.stage_shapes = stage_shapes


        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape) for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(self.input_dims)

        

    # Return main or target actions for given observation and goal
    def get_actions(self, o, g, use_target_net=False):

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]

        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        #print('self.logs():', self.logs())

        while len(ret) == 1:
            ret = ret[0]
        return ret


    # Return Q-Values for given observation, goal and policy action
    def get_Q_values_pi(self, o, g, u, use_target_net=False):

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.Q_pi_tf]

        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    # Return Q-Values for given observation, goal and action taken
    def get_Q_values_u(self, o, g, u, use_target_net=False):

        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.Q_tf]

        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: u.reshape(-1, self.dimu)
        }

        ret = self.sess.run(vals, feed_dict=feed)

        if len(ret) == 1:
            return ret[0]
        else:
            return ret


    # Get stats of normalizers
    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]

        if prefix != '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs


    # Sample batch from my new transitions buffer
    def sample_batch_transitions_buffer(self):

        transitions = self.transitions_buffer.sample(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        transitions['o'], transitions['g'] = self._preprocess_og(o, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, g)

        self.o_stats_mean = np.mean(self.sess.run([self.o_stats.mean]))
        self.o_stats_std = np.mean(self.sess.run([self.o_stats.std]))
        self.g_stats_mean = np.mean(self.sess.run([self.g_stats.mean]))
        self.g_stats_std = np.mean(self.sess.run([self.g_stats.std]))

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]

        return transitions_batch


    # Sample batch from my new transitions buffer
    def sample_batch_erb(self):

        transitions = self.erb.sample(self.batch_size)

        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        transitions['o'], transitions['g'] = self._preprocess_og(o, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, g)

        self.o_stats_mean = np.mean(self.sess.run([self.o_stats.mean]))
        self.o_stats_std = np.mean(self.sess.run([self.o_stats.std]))
        self.g_stats_mean = np.mean(self.sess.run([self.g_stats.mean]))
        self.g_stats_std = np.mean(self.sess.run([self.g_stats.std]))

        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch


    def stage_batch(self, batch=None):
        if batch is None:
            if self.buffer_type == "transitions":
                batch = self.sample_batch_transitions_buffer()  
            elif self.buffer_type == "erb":
                batch = self.sample_batch_erb()  
            else:
                assert False       
                
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))


    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss


    def update_target_net(self):
        self.sess.run(self.update_target_net_op)


    def _create_network(self, input_dims, name=None):
        print("Creating a DDPG agent with max_u {max_u} and u_offset {u_offset} ...".format(max_u=self.max_u, u_offset=self.u_offset))

        # running averages
        with tf.variable_scope('o_stats') as vs:
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)

        with tf.variable_scope('g_stats') as vs:
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])


        # Create networks
        with tf.variable_scope('main'):
            self.main = ActorCritic(batch_tf, net_type='main', **self.__dict__)
            vs.reuse_variables()
        with tf.variable_scope('target'):
            target_batch_tf = batch_tf.copy()
            target_batch_tf['o'] = batch_tf['o_2']
            target_batch_tf['g'] = batch_tf['g_2']
            self.target = ActorCritic(
                target_batch_tf, net_type='target', **self.__dict__)

        assert len(self._vars("main")) == len(self._vars("target"))

        # loss functions
        target_Q_pi_tf = self.target.Q_pi_tf
        clip_range = (-self.clip_return, 0.)
        target_tf = tf.clip_by_value(batch_tf['r'] + self.gamma * target_Q_pi_tf, *clip_range)
        self.Q_loss_tf = tf.reduce_mean(tf.square(tf.stop_gradient(target_tf) - self.main.Q_tf))


        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))

        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run(session=self.sess)
        self._init_target_net()

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res


    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])

        self.critic_loss = critic_loss
        self.actor_loss = actor_loss
        return critic_loss, actor_loss, Q_grad, pi_grad


    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)


    def _preprocess_og(self, o, g):
            o = np.clip(o, -self.clip_obs, self.clip_obs)
            g = np.clip(g, -self.clip_obs, self.clip_obs)
            return o, g


    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)


    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res




class ActorCritic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.
        
        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            # Added the option for offsets to actor network
            self.pi_tf = self.u_offset + self.max_u * tf.tanh(nn(
                input_pi, [self.hidden] * self.layers + [self.dimu]))
        with tf.variable_scope('Q'):
            # for policy training
            input_Q = tf.concat(axis=1, values=[o, g, self.pi_tf / self.max_u])
            self.Q_pi_tf = nn(input_Q, [self.hidden] * self.layers + [1])
            # for critic training
            input_Q = tf.concat(axis=1, values=[o, g, self.u_tf / self.max_u])
            self._input_Q = input_Q  # exposed for tests
            self.Q_tf = nn(input_Q, [self.hidden] * self.layers + [1], reuse=True)

