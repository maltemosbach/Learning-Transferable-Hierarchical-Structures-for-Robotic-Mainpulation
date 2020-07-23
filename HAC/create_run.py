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
        save_new_low_layer = False
        if not batch % TEST_FREQ == 0:
            agent.save_model(batch)
            if (agent.hparams["env"] == "FetchPush-v1" or agent.hparams["env"] == "FetchPickAndPlace-v1") and save_new_low_layer == True:
                agent.save_lowest_layer(batch)
           
        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:
            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            success_rate_plt[ind] = success_rate/100
            writer.add_scalar("success_rate", success_rate/100, ind)
            Critic_losses = agent.log_tb(ind)
            critic_loss_layer0[ind] = Critic_losses[0]
            agent.layers[0].current_sr = success_rate/100
            if agent.hparams["layers"] > 1:
                critic_loss_layer1[ind] = Critic_losses[1]
                agent.layers[1].current_sr = success_rate / 100

            ind += 1
            agent.FLAGS.test = False

            print("\n--- END TESTING ---\n")

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


    return np.copy(success_rate_plt), np.copy(critic_loss_layer0), np.copy(critic_loss_layer1)