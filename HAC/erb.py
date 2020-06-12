import numpy as np
import time


class ExperienceReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, replay_k, reward_fun, replay_strategy='future'):
        """Buffer that stores episodes of varying lenght
        Args:
            buffer_shapes (dict of ints): the shape of all arguments of the buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            replay_k (int): number of HER transitions for every regular transition
            reward_fun (function): reward function
            sampling_strategy (str): HER sampling strategy (future, final or next)
        """
        self.buffer_shapes = buffer_shapes
        self.size_in_transitions = size_in_transitions
        self.reward_fun = reward_fun
        self.replay_strategy = replay_strategy

        # Should include o, ag, g, u, o_2, ag_2
        # {key: array(transition_number, key_shape)}
        self.buffer = {key: np.empty([self.size_in_transitions, buffer_shapes[key]])
                                    for key in buffer_shapes.keys()}

        self.episode_start_idxs = []
        self.episode_start_idxs.append(0)

        # memory management
        self.current_size = 0
        self.size = size_in_transitions
        self.img_size = 0

        assert replay_k >= 0
        self.replay_k = replay_k


    # Resets storing index
    def clear_buffer(self):
        self.current_size = 0


    # Gives current size in transitions
    def get_current_size(self):
        return self.current_size


    def get_img_size(self):
        return self.img_size




    # Sample (batch_size) transitions from the buffer using specified replay strategy
    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """

        #print("Sampling from ERB!")

        assert self.current_size > 0

        if self.replay_strategy == 'future':
            future_p = 1 - (1. / (1 + self.replay_k))
        else:  # 'replay_strategy' == 'none'
            future_p = 0

        transitions = {}
        t_samples = np.random.randint(self.current_size, size=batch_size)

        #print("t_samples:", t_samples)

        transitions = {key: self.buffer[key][t_samples].copy()
                       for key in self.buffer.keys()}

        #print("transitions:", transitions)

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

        her_indexes = her_indexes[0]
        #print("her_indexes:", her_indexes)

        #print("self.episode_start_idxs:", self.episode_start_idxs)

        #print("t_samples[her_indexes]:", t_samples[her_indexes])

        # Calculate future offsets
        future_offset = np.zeros(her_indexes.shape)

        for i in range(her_indexes.shape[0]):
            future_offset[i] = np.random.uniform() * (transitions['ep_ending_idx'][her_indexes][i] - t_samples[her_indexes][i])
            future_offset = future_offset.astype(int)

        #print("future_offset:", future_offset)
        
        future_t = t_samples[her_indexes] + future_offset
        #print("future_t:", future_t)

        future_ag = self.buffer['ag_2'][future_t]
        #print("future_ag:", future_ag)
        transitions['g'][her_indexes] = future_ag

        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = self.reward_fun(**reward_params)

        
        penalize_idxs = np.where(transitions['penalize_sg'] == 0)[0]
        #print("penalize_idxs:", penalize_idxs)
        transitions['r'][penalize_idxs] = -10
        #print("transitions:", transitions)

        # terminal information added
        transitions['is_t'] = np.empty(batch_size, dtype=bool)
        for l in range(batch_size):
            transitions['is_t'][l] = bool(transitions['r'][l] == 0)

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        return transitions


    # Store an episode of any length in the buffer
    def store_episode(self, episode_batch):
        """episode_batch: dict{key: array(timesteps x dim_key)}
        """
        #print("Storing episode in ERB!")

        # Batch includes o, ag, g, u
        episode_batch = {key: episode_batch[key][0] for key in episode_batch.keys()}
        batch_sizes = {key: episode_batch[key].shape for key in episode_batch.keys()}
        T = batch_sizes['u'][0]

        # Creating new observation and new achieved goal
        episode_batch['o_2'] = episode_batch['o'][1:, :]
        episode_batch['ag_2'] = episode_batch['ag'][1:, :]

        # Removing last o_2 from inital observations and same for ag
        episode_batch['o'] = episode_batch['o'][:-1, :]
        episode_batch_save = {}
        episode_batch_save['ag'] = episode_batch['ag']
        episode_batch['ag'] = episode_batch['ag'][:-1, :]

        for key in episode_batch.keys():
            assert episode_batch[key].shape[0] == T

        # Raw episode batch finished here with o, ag, g, u, o_2, ag_2

        episode_batch['ep_ending_idx'] = np.ones((T,1))
        # Get indexes where to store transitions
        idxs = self._get_storage_idx(T)

        # set new episode starting idx
        if np.issubdtype(type(idxs), np.integer):
            self._set_new_start_idx(idxs + 1) 
        else:
            self._set_new_start_idx(idxs[-1] + 1)

        episode_batch['ep_ending_idx'] = (self.episode_start_idxs[-1]-1) * np.ones((T,1))
        episode_batch['penalize_sg'] = episode_batch['penalize_sg'].reshape(T,1)

            
        #print("episode_batch:", episode_batch)

        # load transitions into buffer
        for key in self.buffer.keys():
            self.buffer[key][idxs] = episode_batch[key]


    


    # Get indexes where to store new transitions
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"


        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)

        else:
            if self.img_size + inc > self.size:
                self.img_size = 0 
            idx = np.arange(self.img_size + 0, self.img_size + inc)
            self.img_size = self.img_size + inc

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


    def _set_new_start_idx(self, index):
        self.episode_start_idxs.append(index)















