import gym
from gym import spaces
import numpy as np
import random

class MultiAgentChaseEnv(gym.Env):
    def __init__(self, k=7, n_bots=2, local_grid_size=5, max_steps=100):
        super(MultiAgentChaseEnv, self).__init__()
        self.k = k
        self.n_bots = n_bots
        self.local_grid_size = local_grid_size
        self.max_steps = max_steps
        self.steps_taken = 0

        # Action space: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        self.action_space = spaces.Discrete(4)

        # Observation space for one agent:
        # - local_grid: (local_grid_size, local_grid_size, 4) (player, bots, points, walls)
        # - global_features: 3 (dist to nearest point, wall, other agent)
        # For bots, "other agent" is the player; for player, it's the nearest bot
        # For multi-bot, you can add more features (e.g., dist to other bots)
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(low=0, high=1, shape=(local_grid_size, local_grid_size, 4), dtype=np.float32),
            "global_features": spaces.Box(low=0, high=k, shape=(3,), dtype=np.float32)
        })

        self.reset()

    def reset(self):
        self.grid = np.zeros((self.k, self.k))
        self.walls = np.zeros((self.k, self.k))
        self.points = np.zeros((self.k, self.k))
        self.player_pos = None
        self.bot_positions = []
        self.steps_taken = 0

        # Place walls
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

        return self._get_obs_for_agents()

    def _get_obs_for_agent(self, agent_idx, agent_pos):
        # agent_idx: 0=player, 1..n_bots=bots
        # agent_pos: (x, y) position of the agent

        # --- Local grid ---
        half = self.local_grid_size // 2
        local_grid = np.zeros((self.local_grid_size, self.local_grid_size, 4))

        for i in range(self.local_grid_size):
            for j in range(self.local_grid_size):
                x = agent_pos[0] + i - half
                y = agent_pos[1] + j - half
                if 0 <= x < self.k and 0 <= y < self.k:
                    # Player channel
                    if agent_idx == 0:  # player sees itself as "player"
                        local_grid[i, j, 0] = (x == agent_pos[0] and y == agent_pos[1])
                    else:  # bots see the player as "player"
                        local_grid[i, j, 0] = (x == self.player_pos[0] and y == self.player_pos[1])
                    # Bots channel
                    if agent_idx == 0:  # player sees all bots
                        for bot_pos in self.bot_positions:
                            if x == bot_pos[0] and y == bot_pos[1]:
                                local_grid[i, j, 1] = 1
                    else:  # bots see other bots (except themselves)
                        for idx, bot_pos in enumerate(self.bot_positions):
                            if idx != agent_idx-1 and x == bot_pos[0] and y == bot_pos[1]:
                                local_grid[i, j, 1] = 1
                    # Points channel
                    local_grid[i, j, 2] = self.points[x, y]
                    # Walls channel
                    local_grid[i, j, 3] = self.walls[x, y]

        # --- Global features ---
        # 1. Distance to nearest point
        min_point_dist = self.k * 2  # large value
        for x in range(self.k):
            for y in range(self.k):
                if self.points[x, y] == 1:
                    dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                    if dist < min_point_dist:
                        min_point_dist = dist

        # 2. Distance to nearest wall
        min_wall_dist = self.k * 2
        for x in range(self.k):
            for y in range(self.k):
                if self.walls[x, y] == 1:
                    dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                    if dist < min_wall_dist:
                        min_wall_dist = dist

        # 3. Distance to nearest other agent
        min_agent_dist = self.k * 2
        if agent_idx == 0:  # player: distance to nearest bot
            for bot_pos in self.bot_positions:
                dist = abs(bot_pos[0] - agent_pos[0]) + abs(bot_pos[1] - agent_pos[1])
                if dist < min_agent_dist:
                    min_agent_dist = dist
        else:  # bot: distance to player
            dist = abs(self.player_pos[0] - agent_pos[0]) + abs(self.player_pos[1] - agent_pos[1])
            min_agent_dist = dist

        global_features = np.array([min_point_dist, min_wall_dist, min_agent_dist], dtype=np.float32)

        return {
            "local_grid": local_grid,
            "global_features": global_features
        }

    def _get_obs_for_agents(self):
        # Get observation for player (agent_idx=0)
        player_obs = self._get_obs_for_agent(0, self.player_pos)
        # Get observation for each bot (agent_idx=1..n_bots)
        bot_obs = [self._get_obs_for_agent(i+1, self.bot_positions[i]) for i in range(self.n_bots)]
        return [player_obs] + bot_obs

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

        # Return observations for all agents, reward, done, info
        # Here, reward is for the player; for multi-agent, you might want separate rewards
        return self._get_obs_for_agents(), reward, done, {}


