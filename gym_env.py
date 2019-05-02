import gym

class GymEnv(object):
    def __init__(self, env_name, seed, adv_fraction=1.0):

        env = gym.envs.make(env_name)
        def_adv = env.adv_action_space.high[0]
        new_adv = def_adv*adv_fraction
        env.update_adversary(new_adv)
        
        self.env = env
        self.env_id = env.spec.id
        
        self.seed = seed
        self.env.seed(self.seed)

        self._observation_space = env.observation_space
        self._pro_action_space = env.pro_action_space
        self._adv_action_space = env.adv_action_space
        self._horizon = env.spec.timestep_limit

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def pro_action_space(self):
        return self._pro_action_space

    @property
    def adv_action_space(self):
        return self._adv_action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if hasattr(self.env, 'monitor'):
            if hasattr(self.env.monitor, 'stats_recorder'):
                recorder = self.env.monitor.stats_recorder
                if recorder is not None:
                    recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return next_obs, reward, done, info

    def render(self):
        self.env.render()