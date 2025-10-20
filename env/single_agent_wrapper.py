
import gymnasium as gym
from pettingzoo import ParallelEnv
import numpy as np

class SingleAgentWrapper(gym.Env):
    """
    Wraps a PettingZoo ParallelEnv to expose it as a single-agent Gymnasium Env.
    
    This wrapper is designed to train a single agent from the multi-agent environment.
    The other agents will follow a predefined policy (e.g., random actions).
    """
    def __init__(self, env: ParallelEnv, agent_id: str):
        """
        Initializes the wrapper.

        Args:
            env: The PettingZoo ParallelEnv environment.
            agent_id: The ID of the agent to be trained.
        """
        super().__init__()
        self._env = env
        self._agent_id = agent_id

        # The action and observation spaces are for the single agent
        self.action_space = self._env.action_space(self._agent_id)
        self.observation_space = self._env.observation_space(self._agent_id)

        self._other_agents = [a for a in self._env.possible_agents if a != self._agent_id]

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the observation for the trained agent.
        """
        # The seed handling might need to be more sophisticated if required
        observations, infos = self._env.reset(seed=seed, options=options)
        return observations[self._agent_id], infos.get(self._agent_id, {})

    def step(self, action):
        """
        Steps the environment with the action for the trained agent.
        The other agents take random actions.
        """
        # Build the actions dictionary for all agents
        actions = {self._agent_id: action}
        for agent in self._other_agents:
            if agent in self._env.agents: # Check if the agent is still active
                actions[agent] = self._env.action_space(agent).sample()

        # Step the environment
        observations, rewards, terminations, truncations, infos = self._env.step(actions)

        # Extract results for the trained agent
        obs = observations.get(self._agent_id)
        reward = rewards.get(self._agent_id, 0)
        terminated = terminations.get(self._agent_id, False)
        truncated = truncations.get(self._agent_id, False)
        
        # If the trained agent is done, the episode is over for the wrapper
        if terminated or truncated:
            # Handle the case where the agent might be removed after its final step
            # In PettingZoo, obs/reward might not be present for a terminated agent
            # We should return the last observation we got.
            # For SB3, the final observation is important.
            # The `infos` dict usually contains it.
            if obs is None and 'final_observation' in infos.get(self._agent_id, {}):
                 obs = infos[self._agent_id]['final_observation']
            # If we still don't have an observation, we might need to use a dummy one
            if obs is None:
                obs = np.zeros(self.observation_space.shape)


        return obs, reward, terminated, truncated, infos.get(self._agent_id, {})

    def render(self, mode='human'):
        """ Renders the environment. """
        return self._env.render()

    def close(self):
        """ Closes the environment. """
        return self._env.close()

