import numpy as np
from matplotlib import font_manager, animation
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体（可选）
font_path = '/System/Library/Fonts/STHeiti Light.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 环境参数
grid_size = 4
goal_state = (3, 3)

actions = ['U', 'D', 'L', 'R']
action_dict = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

def get_reward(state):
    return 10 if state == goal_state else 0

def move(state, action):
    x, y = state
    dx, dy = action_dict[action]
    new_x, new_y = x + dx, y + dy
    if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
        return (new_x, new_y)
    else:
        return state

# Q-learning 参数
q_table = np.zeros((grid_size, grid_size, len(actions)))
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 100  # 少一点动画会更快
snapshot_interval = 10

q_table_snapshots = []
episode_paths = []  # 新增路径记录

# 训练 loop
episode_trajectories = []  # 每个 episode 的每一步状态

for episode in range(50):
    state = (1, 1)  # 别用 (0,0)，容易卡墙
    trajectory = [state]

    for step in range(50):
        if np.random.rand() < 1.0:  # 完全探索，确保能动
            action_index = np.random.choice(len(actions))
        else:
            action_index = np.argmax(q_table[state[0], state[1]])

        action = actions[action_index]
        new_state = move(state, action)
        reward = get_reward(new_state)

        old_value = q_table[state[0], state[1], action_index]
        future_max = np.max(q_table[new_state[0], new_state[1]])
        q_table[state[0], state[1], action_index] = old_value + alpha * (reward + gamma * future_max - old_value)

        trajectory.append(new_state)
        state = new_state
        if state == goal_state:
            break

    if episode % 10 == 0:
        print(f"Episode {episode}: Path recorded - {trajectory}")
    episode_trajectories.append(trajectory)

    


# 生成 agent 路径动画
def animate_episodes_step_by_step(trajectories, grid_size, goal_state):
    frames = []
    for ep_idx, path in enumerate(trajectories):
        for step_idx, state in enumerate(path):
            frames.append((ep_idx, step_idx, state, path[0]))

    if len(frames) == 0:
        print("⚠️ 没有可用帧，Agent 可能没学会动。动画未生成。")
        return

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame_idx):
        ax.clear()
        ep, step, curr, start = frames[frame_idx]
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        ax.plot(goal_state[1], goal_state[0], marker='*', markersize=20, color='green', label='Goal')
        ax.plot(start[1], start[0], marker='s', markersize=20, color='blue', label='Start')
        ax.plot(curr[1], curr[0], marker='o', markersize=20, color='red', label='Current')

        ax.set_title(f"Episode {ep}, Step {step}")
        if frame_idx == 0:
            ax.legend(loc='upper right')

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
    ani.save("episode_step_by_step.gif", writer="pillow")
    print("✅ 动画已保存为 episode_step_by_step.gif")


animate_episodes_step_by_step(episode_trajectories, grid_size, goal_state)

