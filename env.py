import gym
from gym import spaces
import numpy as np
import random

class ChaseEnv(gym.Env):
    def __init__(self, k=5, n_bots=1, max_steps=100):
        super(ChaseEnv, self).__init__()
        self.k = k
        self.n_bots = n_bots
        self.max_steps = max_steps
        self.steps_taken = 0

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)
        # For bots, you'd typically have a separate action space, but here we assume same for simplicity

        # Observation space: (k, k, 4) (player, bots, points, walls)
        # Alternatively, use a flat vector or dict observation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(k, k, 4), dtype=np.float32
        )

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

    def step(self, action):
        # Player moves
        new_pos = self._move(self.player_pos, action)
        if self._is_valid_move(new_pos):
            self.player_pos = new_pos

        # Bots move (simple chase: move toward player)
        new_bot_positions = []
        for bot_pos in self.bot_positions:
            # Simple heuristic: move toward player
            bx, by = bot_pos
            px, py = self.player_pos
            dx = 1 if px > bx else -1 if px < bx else 0
            dy = 1 if py > by else -1 if py < by else 0
            # Randomly choose x or y move if both possible (to avoid getting stuck)
            if dx != 0 and dy != 0:
                if random.random() < 0.5:
                    dy = 0
                else:
                    dx = 0
            new_bot_pos = (bx + dx, by + dy)
            if self._is_valid_move(new_bot_pos):
                bot_pos = new_bot_pos
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

        return self._get_obs(), reward, done, {}

# Example usage
# env = ChaseEnv(k=5, n_bots=2)
# obs = env.reset()
# print(obs.shape)  # Should be (5, 5, 4)
