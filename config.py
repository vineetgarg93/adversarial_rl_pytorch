class Config(object):
    """docstring for Config"""
    def __init__(self, env_name = "Reacher-v1"):
        super(Config, self).__init__()

        self.gamma = 0.995
        self.env_name = env_name
        self.tau = 0.97
        self.l2_reg = 1e-3
        self.max_kl = 1e-2
        self.damping = 1e-1
        self.seed = 543
        self.batch_size = 15000
        self.render = False
        self.log_interval = 1