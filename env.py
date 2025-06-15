import gym
from gym import spaces
import numpy as np
import random

class MultiAgentChaseEnv(gym.Env):
    def __init__(self, k=5, n_bots=2, max_steps=100):
        super(MultiAgentChaseEnv, self).__init__()
        self.k = k
        self.n_bots = n_bots
        self.max_steps = max_steps
        self.steps_taken = 0

        # For OpenAI Gym compatibility, define a single agent's action space
        # (This is a bit of a simplification; in practice, you might want to use a Dict or Tuple space)
        self.action_space = spaces.Discrete(4)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        # Observation space: (k, k, 4) (player, bots, points, walls)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(k, k, 4), dtype=np.float32
        )

        # For multi-agent, we'll use a list of actions (one per agent)
        # In reality, each agent should have its own policy, but here we'll assume they all use the same action space

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.k, self.k))
        self.walls = np.zeros((self.k, self.k))
        self.points = np.zeros((self.k, self.k))
        self.player_pos = None
        self.bot_positions = []
        self.steps_taken = 0

        # Place walls (obstacles)
        n_walls = random.randint(1, self.k)
        for _ in range(n_walls):
            x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
            self.walls[x, y] = 1

        # Place points
        n_points = random.randint(1, self.k)
        for _ in range(n_points):
            while True:
                x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
                if self.walls[x, y] == 0:
                    self.points[x, y] = 1
                    break

        # Place player
        while True:
            x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
            if self.walls[x, y] == 0 and self.points[x, y] == 0:
                self.player_pos = (x, y)
                break

        # Place bots
        for _ in range(self.n_bots):
            while True:
                x, y = random.randint(0, self.k-1), random.randint(0, self.k-1)
                if (x, y) != self.player_pos and self.walls[x, y] == 0 and self.points[x, y] == 0:
                    self.bot_positions.append((x, y))
                    break

        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.k, self.k, 4))
        # Player
        x, y = self.player_pos
        obs[x, y, 0] = 1
        # Bots
        for (x, y) in self.bot_positions:
            obs[x, y, 1] = 1
        # Points
        obs[:, :, 2] = self.points
        # Walls
        obs[:, :, 3] = self.walls
        return obs

    def _get_obs_for_agent(self, agent_idx):
        # For the player (agent_idx=0), the observation is the same as global
        # For bots (agent_idx=1..n_bots), you might want to customize the observation
        # Here, we return the same observation for all agents for simplicity
        return self._get_obs()

    def _is_valid_move(self, pos):
        x, y = pos
        return (0 <= x < self.k and 0 <= y < self.k and self.walls[x, y] == 0)

    def _move(self, pos, action):
        x, y = pos
        if action == 0:   # UP
            x = max(0, x-1)
        elif action == 1: # DOWN
            x = min(self.k-1, x+1)
        elif action == 2: # LEFT
            y = max(0, y-1)
        elif action == 3: # RIGHT
            y = min(self.k-1, y+1)
        return (x, y)

    def step(self, actions):
        """
        Args:
            actions: List of actions, one for each agent.
                    actions[0] is the player's action.
                    actions[1..n_bots] are the bots' actions.
        """
        if len(actions) != self.n_bots + 1:
            raise ValueError(f"Expected {self.n_bots + 1} actions, got {len(actions)}")

        # Player moves first
        new_pos = self._move(self.player_pos, actions[0])
        if self._is_valid_move(new_pos):
            self.player_pos = new_pos

        # Bots move (each bot uses its own action)
        new_bot_positions = []
        for i in range(self.n_bots):
            bot_pos = self.bot_positions[i]
            new_bot_pos = self._move(bot_pos, actions[i+1])
            if self._is_valid_move(new_bot_pos):
                new_bot_positions.append(new_bot_pos)
            else:
                new_bot_positions.append(bot_pos)
        self.bot_positions = new_bot_positions

        # Check for point collection
        x, y = self.player_pos
        if self.points[x, y] == 1:
            self.points[x, y] = 0
            point_reward = 1
        else:
            point_reward = 0

        # Check if player caught
        caught = any(bot_pos == self.player_pos for bot_pos in self.bot_positions)
        done = caught or np.sum(self.points) == 0 or self.steps_taken >= self.max_steps

        # Reward
        reward = point_reward
        if caught:
            reward = -10

        self.steps_taken += 1

        # Return observation, reward, done, info
        # For multi-agent, you might return a list of observations and rewards
        # Here, we return the global observation for simplicity
        return self._get_obs(), reward, done, {}

    # Optional: Helper to get observations for each agent
    def get_obs_for_agents(self):
        return [self._get_obs_for_agent(i) for i in range(self.n_bots + 1)]

