import agent as Agent
from tensorboardX import SummaryWriter
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
import numpy as np



def create_run(FLAGS,env,agent,writer,sess, NUM_BATCH):
    """Script that performs one run alternating between training and testing the agent unless 
    specified differently.
        Args:
            FLAGS: flags determining how the algorithm is run
            env: environment the agent is run on
            agent: the HAC agent
            writer: writer for tensorboard logging
            sess: TensorFlow session
            NUM_BATCH (int): total number of batches to be run
        """

    TEST_FREQ = 2
    num_test_episodes = FLAGS.num_test_episodes

    success_rate_plt = np.zeros(np.ceil(NUM_BATCH/2).astype(int))
    critic_loss_layer0 = -1*np.ones(np.ceil(NUM_BATCH/2).astype(int))
    critic_loss_layer1 = -1*np.ones(np.ceil(NUM_BATCH/2).astype(int))
    ind = 0


    Q_VAL_SAMPLING_FREQ = 5
    # Create Q_val_tables (step, layer (0,1), x-dim (10), y-dim (14))
    Q_val_table = np.ones((np.floor(NUM_BATCH/Q_VAL_SAMPLING_FREQ).astype(int)+1, 2, 20, 28))
    

    
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True
     
    for batch in range(NUM_BATCH):

        num_episodes = FLAGS.num_exploration_episodes
        
        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes            

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):
            
            print("\nBatch %d, Episode %d" % (batch, episode))
            
            # Train for an episode
            success = agent.train(env, episode)

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                
                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1            

        # Save agent
        agent.save_model(episode)
        agent.save_lowest_layer(episode)
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:
            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            success_rate_plt[ind] = success_rate/100
            writer.add_scalar("success_rate", success_rate/100, ind)
            Critic_losses = agent.log_tb(ind)
            critic_loss_layer0[ind] = Critic_losses[0]
            if agent.hparams["layers"] > 1:
                critic_loss_layer1[ind] = Critic_losses[1]

            ind += 1
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")

        # Create Q-function matrix if it is the first or last batch
        if batch % Q_VAL_SAMPLING_FREQ == 0:
            Q_vals_layer_0 = np.ones((20, 28))
            Q_vals_layer_1 = np.ones((20, 28))

            # - - - - Q-vals for FetchReach - - - - 
            # Goal is placed near the top left of the plane. For layer 0 the possible states in the plane are evaluated.
            # For layer 1 the position of the gripper is the closer to the bottom right and the actions (subgoals) in the plane
            # are evaluated.
            if env.name == "FetchReach-v1":
                g = np.array([1.345, 0.73, 0.45])
                o = np.zeros([20, 28, 10])

                for i in range(20):
                    for j in range(28):
                        o[i, j, :] = np.array([1.0625 + i*0.025, 0.4125 + j*0.025, 0.5,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        Q_vals_layer_0[i, j] = agent.layers[0].policy.get_Q_values_pi(o[i, j, :], g, np.array([0, 0, 0, 0]), use_target_net=True)

                if agent.hparams["layers"] > 1:
                    g = np.array([1.45, 0.95, 0.45])
                    o = np.array([1.21, 0.57, 0.42,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    u = np.empty((20, 28, 3))

                    for i in range(20):
                        for j in range(28):
                            u[i, j, :] = np.array([1.0625 + i*0.025, 0.4125 + j*0.025, 0.5])
                            if agent.layers[1].policy is not None:
                                Q_vals_layer_1[i, j] = agent.layers[1].policy.get_Q_values_u(o, g, u[i, j, :], use_target_net=True)
                            elif agent.layers[1].critic is not None and agent.hparams["modules"][1] == "TD3":
                                Q_vals_layer_0[i, j] = agent.layers[1].critic.get_target_Q_value_1(np.reshape(o,(1,10)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))
                                Q_vals_layer_1[i, j] = agent.layers[1].critic.get_target_Q_value_2(np.reshape(o,(1,10)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))
                            elif agent.layers[1].critic is not None and agent.hparams["modules"][1] == "actorcritic":
                                Q_vals_layer_1[i, j] = agent.layers[1].critic.get_target_Q_value(np.reshape(o,(1,10)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))


            # - - - - Q-vals for FetchPush and FetchPickAndPlace - - - - 
            elif env.name == "FetchPush-v1" or env.name == "FetchPush_obstacle-v1" or env.name == "FetchPush_obstacle-v2" or env.name == "FetchPickAndPlace-v1" or env.name == "FetchPickAndPlace_obstacle-v1" or env.name == "FetchPickAndPlace_obstacle-v2":
                g = np.array([1.15, 0.6, 0.5])
                o = np.zeros([20, 28, 25])

                for i in range(20):
                    for j in range(28):
                        o[i, j, :] = np.array([1.5, 1.0, 0.45, 1.0625 + i*0.025, 0.4125 + j*0.025, 0.45,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        Q_vals_layer_0[i, j] = agent.layers[0].policy.get_Q_values_pi(o[i, j, :], g, np.array([0, 0, 0, 0]), use_target_net=False)

                if agent.hparams["layers"] > 1:
                    o = np.array([1.5, 1.0, 0.45,  1.4, 0.9, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    u = np.empty((20, 28, 3))        
                    for i in range(20):
                        for j in range(28):
                            u[i, j, :] = np.array([1.0625 + i*0.025, 0.4125 + j*0.025, 0.45])
                            if agent.layers[1].policy is not None:
                                Q_vals_layer_1[i, j] = agent.layers[1].policy.get_Q_values_u(o, g, u[i, j, :], use_target_net=False)
                            elif agent.layers[1].critic is not None:
                                Q_vals_layer_0[i, j] = agent.layers[1].critic.get_target_Q_value_1(np.reshape(o,(1,25)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))
                                Q_vals_layer_1[i, j] = agent.layers[1].critic.get_target_Q_value_2(np.reshape(o,(1,25)), np.reshape(g,(1,3)), np.reshape(u[i, j, :],(1,3)))

            Q_val_table[(batch//Q_VAL_SAMPLING_FREQ), 0, :, :] = Q_vals_layer_0
            Q_val_table[(batch//Q_VAL_SAMPLING_FREQ), 1, :, :] = Q_vals_layer_1



    if FLAGS.play:
        input("Play the trained agent ...")
        agent.FLAGS.show = True
        agent.FLAGS.test = True
        env.visualize = True
        try:
            while True:
                episode = 0
                print("\nEpisode %d" % (episode))
            
                # Test for an episode
                success = agent.train(env, episode)
                episode += 1

                if success:
                    print("Episode %d End Goal Achieved\n" % (episode))

        except KeyboardInterrupt:
            pass


    return np.copy(success_rate_plt), np.copy(Q_val_table), np.copy(critic_loss_layer0), np.copy(critic_loss_layer1)