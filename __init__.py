from gym.envs.registration import register

# using max_episode_steps for new API

register(
    id='rldt-v0',
    entry_point='rldt.envs:DTLearner',
)
