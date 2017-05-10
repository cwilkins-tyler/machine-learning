class Environment(object):
    """Environment within which all agents operate."""
    def __init__(self, verbose=False):
        self.verbose = verbose  # If debug output should be given
        self.grid_width = 11
        self.grid_height = 11

    def create_agent(self, agent_class, *args, **kwargs):
        """ When called, create_agent creates an agent in the environment. """

        agent = agent_class(self, *args, **kwargs)
        return agent
