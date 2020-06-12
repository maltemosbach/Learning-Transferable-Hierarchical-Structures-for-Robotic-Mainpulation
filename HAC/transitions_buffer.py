import numpy as np
import time


class TransitionsBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, replay_k, reward_fun, sampling_strategy="HAC"):
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
        self.sampling_strategy = sampling_strategy

        # Should include o, g, u, r, o_2, is_t
        # {key: array(transition_number, key_shape)}
        self.buffer = {key: np.empty([self.size_in_transitions, buffer_shapes[key]])
                                    for key in buffer_shapes.keys()}

        self.buffer['is_t'] = np.empty([self.size_in_transitions, 1], dtype=bool)

        # memory management
        self.current_size = 0
        self.size = size_in_transitions

        assert replay_k >= 0
        self.replay_k = replay_k


    # Resets storing index
    def clear_buffer(self):
        self.current_size = 0


    # Gives current size in transitions
    def get_current_size(self):
        return self.current_size


    # Add subgoal penalty transition to buffer
    def penalize_subgoal(self, transition):
        """transition: dict{key: array(1 x dim_key)}
        """

        # Get indexes where to store transitions
        idxs = self._get_storage_idx(1)

        # load inputs into buffers
        for key in self.buffer.keys():
            self.buffer[key][idxs] = transition[key]


    # Sample (batch_size) transitions from the buffer
    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        assert self.current_size > 0

        transitions = {}
        t_samples = np.random.randint(self.current_size, size=batch_size)
        
        for key in self.buffer.keys():
            transitions[key] = self.buffer[key][t_samples]

        transitions['r'] = transitions['r'].reshape(batch_size,)
        transitions['is_t'] = transitions['is_t'].reshape(batch_size,)

        return transitions


    # Store an episode of any length in the buffer
    def store_episode(self, episode_batch, goal_status):
        """episode_batch: dict{key: array(timesteps x dim_key)}
        """

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


        sampling_strategy = self.sampling_strategy
        replay_k = self.replay_k

        if sampling_strategy == 'mix':
            if goal_status:
                sampling_strategy = 'final'
            else:
                sampling_strategy = 'HAC'


        if sampling_strategy == 'future':
            # Calculate number of total transitions to store
            total_num_new_trans = (1+replay_k) * T

            # Generate transitions by repeating episode batch to generate regular and HER transitions
            transitions = {key: np.repeat(episode_batch[key], 1 + replay_k, axis=0) for key in episode_batch.keys()}

            indexes = np.arange((1 + replay_k) * T)

            # timestep of each transition inside the episode
            timesteps = np.repeat(np.arange(T), 1 + replay_k)

            # indexes for which HER transitions will be formed
            her_indexes = np.where(np.arange((1 + replay_k) * T) % (1+ replay_k) != 0)

            # offset in timesteps for 'future' sampling strategy
            future_offset = np.random.uniform(size=total_num_new_trans) * (T - timesteps)
            future_offset = future_offset.astype(int)

            # timesteps from which achieved goals will be sampled
            ag_t = (timesteps + 1 + future_offset)[her_indexes]


        elif sampling_strategy == 'final':
            replay_k = 1
            # Calculate number of total transitions to store
            total_num_new_trans = (1+replay_k) * T

            # Generate transitions by repeating episode batch to generate regular and HER transitions
            transitions = {key: np.repeat(episode_batch[key], 1 + replay_k, axis=0) for key in episode_batch.keys()}

            indexes = np.arange((1 + replay_k) * T)

            # timestep of each transition inside the episode
            timesteps = np.repeat(np.arange(T), 1 + replay_k)

            # indexes for which HER transitions will be formed
            her_indexes = np.where(np.arange((1 + replay_k) * T) % (1+ replay_k) != 0)

            # timesteps from which achieved goals will be sampled
            ag_t = (T * np.ones(total_num_new_trans)[her_indexes]).astype(int)


        elif sampling_strategy == 'HAC':
            # Calculate number of total transitions to store
            total_num_new_trans = (1+replay_k) * T

            # Generate transitions by repeating episode batch to generate regular and HER transitions
            transitions = {key: np.repeat(episode_batch[key], 1 + replay_k, axis=0) for key in episode_batch.keys()}

            indexes = np.arange((1 + replay_k) * T)

            # timestep of each transition inside the episode
            timesteps = np.repeat(np.arange(T), 1 + replay_k)
            #print("timesteps:", timesteps)

            # indexes for which HER transitions will be formed
            her_indexes = np.where(np.arange((1 + replay_k) * T) % (1+ replay_k) != 0)

            indices = np.zeros((replay_k))
            indices[:replay_k-1] = np.random.randint(T,size=replay_k-1)
            indices[replay_k-1] = T - 1
            indices = np.sort(indices)

            ag_t = np.tile(indices, T).astype(int)
            ag_t = (ag_t + np.ones(ag_t.shape)).astype(int)


        elif sampling_strategy == 'future-final':
            # Calculate number of total transitions to store
            total_num_new_trans = (1+replay_k) * T

            # Generate transitions by repeating episode batch to generate regular and HER transitions
            transitions = {key: np.repeat(episode_batch[key], 1 + replay_k, axis=0) for key in episode_batch.keys()}

            indexes = np.arange((1 + replay_k) * T)

            # timestep of each transition inside the episode
            timesteps = np.repeat(np.arange(T), 1 + replay_k)
            #print("timesteps:", timesteps)

            # indexes for which HER transitions will be formed
            her_indexes = np.where(np.arange((1 + replay_k) * T) % (1+ replay_k) != 0)
            #print("her_indexes:", her_indexes)

            indices = np.zeros((replay_k))
            indices[:replay_k-1] = np.random.randint(T,size=replay_k-1)
            indices[replay_k-1] = T - 1
            indices = np.sort(indices)
            
            # offset in timesteps
            future_offset = np.random.uniform(size=total_num_new_trans) * (T - timesteps)
            future_offset = future_offset.astype(int)
            finals = future_offset[replay_k::1+replay_k]

            finals[:] = np.flip(np.arange(T),0)

            ag_t = (timesteps + 1 + future_offset)[her_indexes]


        elif sampling_strategy == 'none':
            replay_k = 0
            # Calculate number of total transitions to store
            total_num_new_trans = (1+replay_k) * T

            # Generate transitions by repeating episode batch to generate regular and HER transitions
            transitions = {key: np.repeat(episode_batch[key], 1 + replay_k, axis=0) for key in episode_batch.keys()}

            indexes = np.arange((1 + replay_k) * T)

            # timestep of each transition inside the episode
            timesteps = np.repeat(np.arange(T), 1 + replay_k)

            # indexes for which HER transitions will be formed
            her_indexes = []

            # timesteps from which future achieved goals will be sampled
            ag_t = (T * np.ones(total_num_new_trans)[her_indexes]).astype(int)

        else:
            assert False, "Unknown sampling strategy."




        # Selecting achieved goals
        ags = episode_batch_save['ag'][ag_t]

        # Substituting goals with selected achieved goals
        transitions['g'][her_indexes] = ags

        # Calculating reward for all transitions
        reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
        transitions['r'] = self.reward_fun(**reward_params)

        # Adding 'is terminal' information to all transitions
        transitions['is_t'] = np.empty(total_num_new_trans, dtype=bool)
        for l in range(total_num_new_trans):
            transitions['is_t'][l] = bool(transitions['r'][l] == 0)

        transitions['r'] = transitions['r'].reshape(total_num_new_trans, 1)
        transitions['is_t'] = transitions['is_t'].reshape(total_num_new_trans, 1)

        # Get indexes where to store transitions
        idxs = self._get_storage_idx(total_num_new_trans)

        # load transitions into buffer
        for key in self.buffer.keys():
            self.buffer[key][idxs] = transitions[key]


    


    # Get indexes where to store new transitions
    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx















