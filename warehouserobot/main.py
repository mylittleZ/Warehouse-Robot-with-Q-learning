import numpy as np
import matplotlib.pyplot as plt

# Define the learning environment
num_rows, num_columns = 11, 11

# Initialize Q-values
q_matrix = np.zeros((num_rows, num_columns, 4))
directions = ['up', 'right', 'down', 'left']

# Define rewards
penalties = np.full((num_rows, num_columns), -100.)
penalties[0, 5] = 100.

# Define open areas
open_areas = {1: list(range(1, 10)), 2: [1, 7, 9], 3: list(range(1, 8)) + [9],
              4: [3, 7], 5: list(range(num_columns)), 6: [5], 7: list(range(1, 10)),
              8: [3, 7], 9: list(range(num_columns))}
for row in range(1, 10):
    for col in open_areas[row]:
        penalties[row, col] = -1.

# Check if state is terminal
def check_terminal_state(row, col):
    return penalties[row, col] != -1.

# Get a random starting position
def random_start():
    start_row, start_col = np.random.randint(num_rows), np.random.randint(num_columns)
    # Ensure the starting location is not a terminal state
    while check_terminal_state(start_row, start_col):
        start_row, start_col = np.random.randint(num_rows), np.random.randint(num_columns)
    return start_row, start_col

# Select action using epsilon-greedy strategy
def select_action(row, col, epsilon_value):
    if np.random.random() < epsilon_value:
        return np.random.randint(4)
    return np.argmax(q_matrix[row, col])

# Calculate next position
def calculate_next_position(row, col, action):
    if directions[action] == 'up' and row > 0:
        row -= 1
    elif directions[action] == 'right' and col < num_columns - 1:
        col += 1
    elif directions[action] == 'down' and row < num_rows - 1:
        row += 1
    elif directions[action] == 'left' and col > 0:
        col -= 1
    return row, col

def find_shortest_path(start_row, start_col):
    if check_terminal_state(start_row, start_col):
        return []  # If the start is a terminal state, return an empty path

    path = [[start_row, start_col]]
    while not check_terminal_state(start_row, start_col):
        action = select_action(start_row, start_col, 0)
        start_row, start_col = calculate_next_position(start_row, start_col, action)
        path.append([start_row, start_col])
    return path


# Training parameters
exploration_rate = 0.1
gamma = 0.9
alpha = 0.9

# Train for 1000 episodes
for episode in range(1000):
    current_row, current_col = random_start()
    while not check_terminal_state(current_row, current_col):
        action = select_action(current_row, current_col, exploration_rate)
        old_row, old_col = current_row, current_col
        current_row, current_col = calculate_next_position(current_row, current_col, action)
        reward = penalties[current_row, current_col]
        old_value = q_matrix[old_row, old_col, action]
        td_error = reward + gamma * np.max(q_matrix[current_row, current_col]) - old_value
        q_matrix[old_row, old_col, action] = old_value + alpha * td_error


# Function to plot path on grid
def plot_path(grid_map, path):
    # Configuration for the plot
    color_map = plt.cm.jet
    norm = plt.Normalize(vmin=-100, vmax=100)
    colors = [[norm(-100), "black"], [norm(-1), "white"], [norm(100), "red"]]
    color_map = plt.cm.colors.LinearSegmentedColormap.from_list("", colors)

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.matshow(grid_map, cmap=color_map, interpolation='nearest')

    # Drawing grid lines
    for x in range(grid_map.shape[1] + 1):
        ax.axhline(x - 0.5, lw=2, color='k', zorder=5)
        ax.axvline(x - 0.5, lw=2, color='k', zorder=5)

    # Adding labels
    ax.set_xticks(np.arange(grid_map.shape[1]))
    ax.set_yticks(np.arange(grid_map.shape[0]))
    ax.set_xticklabels(np.arange(grid_map.shape[1]))
    ax.set_yticklabels(np.arange(grid_map.shape[0]))

    # Annotating rewards
    for i in range(grid_map.shape[0]):
        for j in range(grid_map.shape[1]):
            ax.text(j, i, str(grid_map[i, j]), va='center', ha='center', color='pink')

    # Plotting the path
    if path:
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            ax.annotate('', xy=(end[1], end[0]), xytext=(start[1], start[0]),
                        arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle="->", linewidth=2))

    plt.show()

# Main execution
grid_layout = np.array(penalties)

# Display path
initial_location = (9, 7)
optimal_path = find_shortest_path(*initial_location)
print("Optimal Path:", optimal_path)
plot_path(grid_layout, optimal_path)

# Reverse path
reverse_path = find_shortest_path(5, 2)
reverse_path.reverse()
print("Reverse Optimal Path:", reverse_path)
plot_path(grid_layout, reverse_path)
