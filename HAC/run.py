
"""
This is the starting file for all runs. The command line options are being processed and the algorithm is executed accordingly.
"""
import multiprocessing
from datetime import datetime
from init_runs import init_runs
from utils import _get_combinations
from options import parse_options
from pathlib import Path

FLAGS = parse_options()



#  #  #  #  #  D E F I N E    A L L    P A R A M E T E R S  #  #  #  #  #


""" 1. HYPERPARAMETERS
The key hyperparameters are:
    env (str): Environment the algorithm should be run on
    ac_n (float): Noise added to the actions during exploration
    sg_n (float): Noise added to the subgoals during exploration
    replay_k (int): Number of HER transitions per regular transition
    layers (int): Number of hierarchical layers in the algorithm (1, 2)
    use_target (array of booleans): Whether each layer should use target networks
    sg_test_perc (float): Percentage of subgoal testing transitions
    buffer (array of strs): Which buffer each layer should use ('experience', 'replay', 'transitions')
    modules (array of strs): Modules each layer should use (baselineDDPG, actorcritic right now)
    tl-mode (str): Mode for transfer-learning, which is only relevant when the --transfer flag is used (shared_LL, separate_LL, shared_LL_noHAT)
"""
hyperparameters = {
        "env"          : ['FetchPickAndPlace_variation2-v1'],
        "ac_n"         : [0.2],
        "sg_n"         : [0.2],
        "replay_k"     : [4],
        "layers"       : [2],
        "use_target"   : [[False, True]],
        "sg_test_perc" : [0.15],
        "buffer"       : [['transitions', 'transitions']],
        "samp_str"     : ['HAC'],
        "modules"      : [['baselineDDPG', 'actorcritic']],
        "tl-mode"      : ['separate_LL']

    }

hparams = _get_combinations(hyperparameters)

""" 2. PARAMETERS FOR RUNS AND TIME-SCALES
Parameters for the runs
    NUM_RUNS (int): Number of runs for each hyperparameter combination
    NUM_BATCH (int): Total number of batches for each run (one batch is made up of 10 (during testing) or 100 (during exploration) full episodes)
    FLAGS.time_scale (int): Max sequence length in which each policy will specialize
    FLAGS.max_actions (int): Max number of atomic actions
    FLAGS.subgoal_penalty (float): Penalty given to higher layer in subgoal testing transitions where the proposed goal is not reached
    FLAGS.num_exploration_episodes (int): Number of training episodes in one batch/ epoch
    FLAGS.num_test_episodes (int): Number of testing episodes after every epoch of training
"""

NUM_RUNS = 1 #multiprocessing.cpu_count() // len(hparams)
NUM_BATCH = 201

#FLAGS.np = multiprocessing.cpu_count()

FLAGS.time_scale = 10
FLAGS.max_actions = 50

FLAGS.subgoal_penalty = -FLAGS.time_scale
FLAGS.num_exploration_episodes = 100
FLAGS.num_test_episodes = 100

FLAGS.sr_based_exploration = False


""" 3. ADDITIONAL OPTIONS
More settings
    save_models (boolean): Whether all models should be saved
"""
save_models = True


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #












if FLAGS.test == False and FLAGS.retrain == False:
    FLAGS.retrain = True

date = datetime.now().strftime("%d.%m-%H:%M")

# Data directory to coordinate the output-data of the different runs
datadir = "./data/" + date
Path(datadir).mkdir(parents=True, exist_ok=True)

for i in range(len(hparams)):
    Path(datadir + "/graph_" + str(i)).mkdir(parents=True, exist_ok=True)



# Testing a trained agent
if FLAGS.test:
    assert len(hparams) == 1, "To test a trained agent only one parameter configuration should be given"
    init_runs(date, hparams[0], 1, None, FLAGS, NUM_BATCH, False)


# Training new agents
if FLAGS.retrain:

    # Run such that one process is used for one hparam combination
    if FLAGS.np == len(hparams):
        print("Running (retraining) with {num_proc} processes ...".format(num_proc=len(hparams)))
        if __name__ == '__main__':
            processes = []
            for i in range(len(hparams)):
                p = multiprocessing.Process(target=init_runs, args=(date, hparams[i], NUM_RUNS, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models, ))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()


    # Run such that each individual run is a process
    elif FLAGS.np == len(hparams)*NUM_RUNS:
        print("Running (retraining) with {num_proc} processes ...".format(num_proc=len(hparams)*NUM_RUNS))
        if __name__ == '__main__':
            processes = []
            for i in range(len(hparams)):
                for j in range(NUM_RUNS):
                    p = multiprocessing.Process(target=init_runs, args=(date, hparams[i], 1, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models, j,))
                    processes.append(p)
                    p.start()

            for process in processes:
                process.join()

    # Run in serial execution
    else:
        if FLAGS.np is not None and FLAGS.np != 1:
            print("Possible number of processes are 1, {n1}, or {n2} but {n3} was given instead. Falling back on serial execution.".format(n1=len(hparams), n2=len(hparams)*NUM_RUNS, n3=FLAGS.np))
        print("Running (retraining) in serial execution ...")
        for i in range(len(hparams)):
            init_runs(date, hparams[i], NUM_RUNS, datadir + "/graph_" + str(i), FLAGS, NUM_BATCH, save_models)
