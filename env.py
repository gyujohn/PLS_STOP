import gym
from gym import spaces
import numpy as np
import random


# 
class MazeEnv(gym.Env):
    def __init__(self, k=11, n_bots=2, max_steps=1000):
        super(MazeEnv, self).__init__()
        
        self.k = k 
        self.local_grid_size = k
        self.n_bots = max(min(n_bots, 4), 4)
        
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


# Creates an env:
# - self.points
# - self.walls
# - self.player_pos
# - self.bot_positions
# 
# Generation Constraints
# - four corners cannot be placed with wall
# 
class EnvGenerator():
    
    
    
    def reset(self):
        
        self.steps_taken = 0
        self.bot_positions = []
        
        self.points = np.zeros((self.k, self.k))
        self.walls = np.ones((self.k, self.k), dtype=int)  # Start fully walled
        
        # Generate maze using randomized DFS
        self._generate_maze()
        
        # Place player on a random empty cell
        empty_cells = list(zip(*np.where(self.walls == 0)))
        self.player_pos = random.choice(empty_cells)
        
        # Place bots on empty cells reachable from player
        reachable = self._bfs_reachable(self.player_pos)
        reachable_cells = [cell for cell in empty_cells if cell in reachable and cell != self.player_pos]
        
        if len(reachable_cells) < self.n_bots:
            raise RuntimeError("Not enough reachable cells to place bots")

        self.bot_positions = random.sample(reachable_cells, self.n_bots)
        
        # Place points on reachable empty cells not occupied by player or bots
        available_cells = [cell for cell in reachable if cell != self.player_pos and cell not in self.bot_positions]
        n_points = random.randint(1, min(len(available_cells), self.k))
        points_cells = random.sample(available_cells, n_points)
        for x, y in points_cells:
            self.points[x, y] = 1
        
        return self._get_obs_for_agents()
    
    def _generate_maze(self):
        """
        Generate maze using randomized DFS.
        Walls = 1, paths = 0.
        Maze size self.k x self.k.
        For simplicity, assume odd dimensions for proper maze carving.
        """
        # Initialize all cells as walls
        self.walls.fill(1)

        # Start DFS from a random odd cell
        start_x = random.randrange(1, self.k, 2)
        start_y = random.randrange(1, self.k, 2)
        self.walls[start_x, start_y] = 0

        stack = [(start_x, start_y)]
        directions = [(2,0), (-2,0), (0,2), (0,-2)]

        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.k-1 and 1 <= ny < self.k-1:
                    if self.walls[nx, ny] == 1:
                        neighbors.append((nx, ny))
            if neighbors:
                nx, ny = random.choice(neighbors)
                # Remove wall between current and neighbor
                wx, wy = (x + nx)//2, (y + ny)//2
                self.walls[wx, wy] = 0
                self.walls[nx, ny] = 0
                stack.append((nx, ny))
            else:
                stack.pop()

    def _bfs_reachable(self, start):
        """
        Return set of reachable cells from start using BFS on free cells (walls=0).
        """
        visited = set()
        queue = deque([start])
        visited.add(start)

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.k and 0 <= ny < self.k:
                    if self.walls[nx, ny] == 0 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return visited

    
    def count_unique_paths_dfs(grid, start=(5,5)):
        """
        Count all unique paths from start to each of the four corners using DFS/backtracking.
        Returns a list: [paths to (0,0), paths to (0,N-1), paths to (N-1,0), paths to (N-1,N-1)]
        """
        N = grid.shape[0]
        corners = [(0,0), (0,N-1), (N-1,0), (N-1,N-1)]
        path_counts = []
    
        for end in corners:
            # If the corner is a wall (should not happen as per your guarantee), skip
            if grid[end] == 1:
                path_counts.append(0)
                continue
    
            # Initialize visited for current path
            visited = np.zeros((N,N), dtype=bool)
            path = []
            all_paths = []
    
            def dfs(x, y):
                if (x, y) == end:
                    # Record the current path (optional: store the path, here we just count)
                    all_paths.append(1)
                    return
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < N and 0 <= ny < N and 
                        grid[nx,ny] == 0 and not visited[nx,ny]):
                        visited[nx,ny] = True
                        dfs(nx, ny)
                        visited[nx,ny] = False
    
            # Start DFS
            visited[start] = True
            dfs(start[0], start[1])
            path_counts.append(len(all_paths))
    
        return path_counts
    
    # Example usage:
    # grid = np.zeros((11, 11))
    # path_counts = count_unique_paths_dfs(grid, (5,5))
    # print(path_counts)  # e.g., [924, 924, 924, 924] for a 11x11 grid with no walls

    
