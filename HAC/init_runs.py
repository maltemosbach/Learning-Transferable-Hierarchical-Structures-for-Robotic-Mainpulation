from environment import Environment
from agent import Agent
from create_run import create_run
from tensorboardX import SummaryWriter
import tensorflow as tf
import os
import shutil

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def init_runs(date, hparams, num_runs, data_dir, FLAGS, NUM_BATCH, save_models, idx_run=0):
    """Script that initializes the runs and organizes the data for multiprocessing.
        Args:
            date (str): timestamp
            hparams: hyperparameters from run.py
            num_runs (int): number of consecutive runs this script should execute
            data_dir (str): path to data directory
            FLAGS: flags determining how the algorithm is run
            NUM_BATCH (int): total number of batches to be run
            save_models (bool): whether the trained agents should be saved
            idx_run (int): index used to organize the data
        """

    for m in range(num_runs):

        if num_runs > 1:
            idx_run = m


        # Directories for models and logging
        hp_dir = "/"
        for arg in hparams:
            if arg != "run":
                hp_dir = hp_dir + arg + "=" + str(hparams[arg]) + "/"
        hp_dir = hp_dir
        logdir = "./tb/" + date + hp_dir + "run_" + str(idx_run)
        modeldir = "./saved_agents/" + hp_dir + "models" + "_" + str(idx_run)

        sess = tf.compat.v1.InteractiveSession()
        writer_graph = tf.compat.v1.summary.FileWriter(logdir)
        writer = SummaryWriter(logdir)


        # Create agent and environment

        # Projection functions from state to endgoal and sub-goals
        if hparams["env"] == "FetchReach-v1":
            project_state_to_end_goal = lambda sim, state: state[0:3]
            project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[0] > 1.55 else 1.05 if state[0] < 1.05 else state[0], 1.1 if state[1] > 1.1 else 0.4 if state[1] < 0.4 else state[1], 1.1 if state[2] > 1.1 else 0.4 if state[2] < 0.4 else state[2]])
        elif hparams["env"] == "FetchPush-v1" or hparams["env"] == "FetchPush_variation1-v1" or hparams["env"] == "FetchPush_variation2-v1" or hparams["env"] == "FetchPickAndPlace-v1" or hparams["env"] == "FetchPickAndPlace_variation1-v1" or hparams["env"] == "FetchPickAndPlace_variation2-v1":
            project_state_to_end_goal = lambda sim, state: state[3:6]
            project_state_to_subgoal = lambda sim, state: np.array([1.55 if state[3] > 1.55 else 1.05 if state[3] < 1.05 else state[3], 1.1 if state[4] > 1.1 else 0.4 if state[4] < 0.4 else state[4], 1.1 if state[5] > 1.1 else 0.4 if state[5] < 0.4 else state[5]])
        else:
            assert False, "Unknown environment given."

        dist_threshold = 0.05
        end_goal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])

        # Provide range for each dimension of subgoal space in order to configure subgoal actor networks.  Subgoal space can be the same as the state space or some other projection out of the state space.
        subgoal_bounds = np.array([[1.05, 1.55], [0.4, 1.1], [0.4, 1.1]])

        # subgoal_thresholds = np.array([len_threshold, len_threshold, height_threshold, quat_threshold, quat_threshold, quat_threshold, quat_threshold, velo_threshold, velo_threshold, velo_threshold])
        subgoal_thresholds = np.array([dist_threshold, dist_threshold, dist_threshold])


        # Instantiate and return agent and environment
        env = Environment(hparams["env"], project_state_to_end_goal, end_goal_thresholds, subgoal_bounds, project_state_to_subgoal, subgoal_thresholds, FLAGS.max_actions, FLAGS.show)
        agent = Agent(FLAGS,env, writer, writer_graph, sess, hparams)


        # Begin training
        success_rates_run, Q_val_table_run, critic_loss_layer0, critic_loss_layer1 = create_run(FLAGS,env,agent,writer,sess, NUM_BATCH)

        if FLAGS.retrain:
            np.save(data_dir + "/sr_run_" + str(idx_run) + ".npy", success_rates_run)
            np.save(data_dir + "/Q_val_table_run_" + str(idx_run) + ".npy", Q_val_table_run)
            np.save(data_dir + "/critic_loss_layer0_run_" + str(idx_run) + ".npy", critic_loss_layer0)
            np.save(data_dir + "/critic_loss_layer1_run_" + str(idx_run) + ".npy", critic_loss_layer1)
            if m == 0:
                with open(data_dir + "/title.txt", 'w') as outfile:
                    outfile.write(hparams["env"])
                with open(data_dir + "/hparams.txt", 'w') as outfile:
                    outfile.write(str(hparams))


        # Saving models 
        if save_models:
            shutil.move("./models", modeldir)


        sess.close()
        tf.compat.v1.reset_default_graph()

    
