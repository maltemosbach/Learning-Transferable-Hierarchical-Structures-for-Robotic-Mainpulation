import numpy as np
from layer import Layer
from environment import Environment
import tensorflow as tf
import os
from tensorboardX import SummaryWriter

import time


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


# Below class instantiates an agent
class Agent():
    def __init__(self, FLAGS, env, writer, writer_graph, sess, hparams):

        self.FLAGS = FLAGS
        self.sess = sess
        self.writer = writer
        self.writer_graph = writer_graph
        self.hparams = hparams

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = hparams["sg_test_perc"]

        # Create agent with number of levels specified by user
        self.layers = [Layer(i, FLAGS, env, self.sess, self.writer, hparams) for i in range(hparams["layers"])]

        # Below attributes will be used help save network parameters
        self.saver = None
        self.saver_lowest_layer = None
        self.model_dir = None
        self.model_dir_lowest_layer = None
        self.model_loc = None
        self.model_loc_lowest_layer = None

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()
        self.writer_graph.add_graph(sess.graph)

        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(hparams["layers"])]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self, env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.hparams["layers"])]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_end_goal = env.project_state_to_end_goal(env.sim, self.current_state)

        for i in range(self.hparams["layers"]):

            goal_achieved = True

            # If at highest layer, compare to end goal thresholds
            if i == self.hparams["layers"] - 1:

                # Check dimensions are appropriate
                assert len(proj_end_goal) == len(self.goal_array[i]) == len(
                    env.end_goal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                if goal_distance(self.goal_array[i], proj_end_goal) > env.end_goal_thresholds[0]:
                    goal_achieved = False
                    break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(
                    env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                if goal_distance(self.goal_array[i], proj_subgoal) > env.subgoal_thresholds[0]:
                    goal_achieved = False
                    break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False

        return goal_status, max_lay_achieved

    def initialize_networks(self):

        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="")
        model_vars_lowest_layer = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="DDPG_layer_0")

        self.saver = tf.compat.v1.train.Saver(model_vars, max_to_keep=1000)
        self.saver_lowest_layer = tf.compat.v1.train.Saver(model_vars_lowest_layer, max_to_keep=1000)

        # Set up directory for saving models
        self.model_dir = os.getcwd() + '/models'
        self.model_dir_lowest_layer = os.getcwd() + '/tl_models/' + self.hparams["env"]
        self.model_loc = self.model_dir + '/HAC.ckpt'
        self.model_loc_lowest_layer = self.model_dir_lowest_layer + '/HAC.ckpt'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.model_dir_lowest_layer) and (self.hparams["env"] == "FetchPush-v1" or self.hparams["env"] == "FetchPickAndPlace-v1"):
            os.makedirs(self.model_dir_lowest_layer)

        # Initialize actor/critic networks
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # If not retraining or transfer learning, restore all weights
        if self.FLAGS.retrain == False and self.hparams["use_tl"] == False and self.FLAGS.contin == False:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

        # For transfer learning, restore lowest layer
        if self.hparams["use_tl"] == True:
            if self.hparams['env'] == 'FetchPush_variation1-v1' or self.hparams['env'] == 'FetchPush_variation2-v1':
                self.saver_lowest_layer.restore(self.sess, tf.train.latest_checkpoint(os.getcwd() + '/tl_models'
                                                                                                    '/FetchPush-v1'))
            elif self.hparams['env'] == 'FetchPickAndPlace_variation1-v1' or \
                    self.hparams['env'] == 'FetchPickAndPlace_variation2-v1':
                self.saver_lowest_layer.restore(self.sess, tf.train.latest_checkpoint(
                    '{0}/tl_models/FetchPickAndPlace-v1'.format(os.getcwd())))

            elif self.hparams['env'] == 'FetchPush-v1':
                print("WARNING: Training on the regular environment with transfer learning!")
                time.sleep(5)
                self.saver_lowest_layer.restore(self.sess, tf.train.latest_checkpoint(os.getcwd() + '/tl_models'
                                                                                                    '/FetchPush-v1'))
            elif self.hparams['env'] == 'FetchPickAndPlace-v1':
                print("WARNING: Training on the regular environment with transfer learning!")
                time.sleep(5)
                self.saver_lowest_layer.restore(self.sess, tf.train.latest_checkpoint(
                    '{0}/tl_models/FetchPickAndPlace-v1'.format(os.getcwd())))

            else:
                assert False, "Error in transfer learning loading of variables!"

        # Continue with all layers
        if self.FLAGS.contin == True:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))

    # Save neural network parameters
    def save_model(self, episode):
        self.saver.save(self.sess, self.model_loc, global_step=episode)

    # Save neural network parameters of lowest layer
    def save_lowest_layer(self, episode):
        self.saver_lowest_layer.save(self.sess, self.model_loc_lowest_layer, global_step=episode)

    # Update actor and critic networks for each layer
    def learn(self):

        for i in range(len(self.layers)):
            self.layers[i].learn(self.num_updates)

    # Train agent for an episode
    def train(self, env, episode_num):

        # Select initial state from in initial state space, defined in environment.py
        self.current_state = env.reset_sim(self.goal_array[self.hparams["layers"] - 1])

        # Select final goal from final goal space, defined in "design_agent_and_env.py"
        self.goal_array[self.hparams["layers"] - 1] = env.get_next_goal(self.FLAGS.test)
        if self.FLAGS.verbose:
            print("Next End Goal: ", self.goal_array[self.hparams["layers"] - 1])

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.hparams["layers"] - 1].train(self, env,
                                                                                      episode_num=episode_num)

        # Update actor/critic networks if not testing
        if not self.FLAGS.test:
            self.learn()

        # Return whether end goal was achieved
        return goal_status[self.hparams["layers"] - 1]

    # Log any variables to tensorboard
    def log_tb(self, step):
        log = True
        ret = []
        if log:
            if self.hparams["modules"][0] == "baselineDDPG":
                if self.hparams["buffer"][0] == "replay":
                    self.writer.add_scalar('layer_0/replay_buffer size',
                                           self.layers[0].replay_buffer.get_current_size(), step)
                elif self.hparams["buffer"][0] == "transitions":
                    self.writer.add_scalar('layer_0/transitions_buffer size',
                                           self.layers[0].transitions_buffer.get_current_size(), step)
                elif self.hparams["buffer"][0] == "erb":
                    self.writer.add_scalar('layer_0/ER_buffer size', self.layers[0].erb.get_current_size(), step)
                    self.writer.add_scalar('layer_0/ER_buffer img_size', self.layers[0].erb.get_img_size(), step)
                elif self.hparams["buffer"][0] == "experience":
                    self.writer.add_scalar('layer_0/experience_buffer size', self.layers[0].experience_buffer.size,
                                           step)

                self.writer.add_histogram('layer_0/actor_loss', self.layers[0].policy.actor_loss, step)
                self.writer.add_scalar('layer_0/critic_loss', self.layers[0].policy.critic_loss, step)

                self.writer.add_scalar('layer_0/stats_o/mean', self.layers[0].policy.o_stats_mean, step)
                self.writer.add_scalar('layer_0/stats_o/std', self.layers[0].policy.o_stats_std, step)
                self.writer.add_scalar('layer_0/stats_g/mean', self.layers[0].policy.g_stats_mean, step)
                self.writer.add_scalar('layer_0/stats_g/std', self.layers[0].policy.g_stats_std, step)

                # Logging exemplary weight to verify transfer learning
                self.writer.add_histogram('layer_0/actor_weights_main',
                                          self.layers[0].policy._vars('main/pi/_0')[0].eval(session=self.sess), step)
                self.writer.add_histogram('layer_0/critic_weights_main',
                                          self.layers[0].policy._vars('main/Q/_0')[0].eval(session=self.sess), step)
                self.writer.add_histogram('layer_0/actor_weights_target',
                                          self.layers[0].policy._vars('target/pi/_0')[0].eval(session=self.sess), step)
                self.writer.add_histogram('layer_0/critic_weights_target',
                                          self.layers[0].policy._vars('target/Q/_0')[0].eval(session=self.sess), step)

                ret.append(self.layers[0].policy.critic_loss)

            if self.hparams["layers"] > 1:
                if self.hparams["buffer"][1] == "replay":
                    self.writer.add_scalar('layer_1/replay_buffer size',
                                           self.layers[1].replay_buffer.get_current_size(), step)
                elif self.hparams["buffer"][1] == "transitions":
                    self.writer.add_scalar('layer_1/transitions_buffer size',
                                           self.layers[1].transitions_buffer.get_current_size(), step)
                elif self.hparams["buffer"][1] == "experience":
                    self.writer.add_scalar('layer_1/experience_buffer size', self.layers[1].experience_buffer.size,
                                           step)

                if self.hparams["modules"][1] == "actorcritic":
                    # Logging exemplary weight to verify transfer learning
                    self.writer.add_histogram('layer_1/critic_weights_main',
                                              self.layers[1].critic.weights[0].eval(session=self.sess), step)
                    self.writer.add_histogram('layer_1/critic_weights_target',
                                              self.layers[1].critic.target_weights[0].eval(session=self.sess), step)
                    self.writer.add_histogram('layer_1/actor_weights_main',
                                              self.layers[1].actor.weights[0].eval(session=self.sess), step)
                    self.writer.add_histogram('layer_1/actor_weights_target',
                                              self.layers[1].actor.target_weights[0].eval(session=self.sess), step)
                    self.writer.add_scalar('layer_1/actor/stats_o/mean', self.layers[1].actor.o_stats_mean, step)
                    self.writer.add_scalar('layer_1/actor/stats_o/std', self.layers[1].actor.o_stats_std, step)
                    self.writer.add_scalar('layer_1/actor/stats_g/mean', self.layers[1].actor.g_stats_mean, step)
                    self.writer.add_scalar('layer_1/actor/stats_g/std', self.layers[1].actor.g_stats_std, step)
                    self.writer.add_scalar('layer_1/critic/stats_o/mean', self.layers[1].critic.o_stats_mean, step)
                    self.writer.add_scalar('layer_1/critic/stats_o/std', self.layers[1].critic.o_stats_std, step)
                    self.writer.add_scalar('layer_1/critic/stats_g/mean', self.layers[1].critic.g_stats_mean, step)
                    self.writer.add_scalar('layer_1/critic/stats_g/std', self.layers[1].critic.g_stats_std, step)
                    self.writer.add_scalar('layer_1/critic/stats_u/mean', self.layers[1].critic.u_stats_mean, step)
                    self.writer.add_scalar('layer_1/critic/stats_u/std', self.layers[1].critic.u_stats_std, step)

                self.writer.add_scalar('layer_1/critic_loss', self.layers[1].critic.loss_val, step)
                ret.append(self.layers[1].critic.loss_val)

        return ret
