import numpy as np
import matplotlib.pyplot as plt

# Environment setup
environment_rows = 11
environment_columns = 11
q_values = np.zeros((environment_rows, environment_columns, 4))
actions = ['up', 'right', 'down', 'left']

# Rewards setup
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100.  # Goal location with high reward
aisles = {1: [i for i in range(1, 10)], 2: [1, 7, 9], 3: [i for i in range(1, 8)] + [9],
          4: [3, 7], 5: [i for i in range(11)], 6: [5], 7: [i for i in range(1, 10)],
          8: [3, 7], 9: [i for i in range(11)]}
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.

# Function definitions
def is_terminal_state(current_row_index, current_column_index):
    return rewards[current_row_index, current_column_index] != -1.

def get_starting_location():
    current_row_index, current_column_index = np.random.randint(environment_rows), np.random.randint(environment_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index, current_column_index = np.random.randint(environment_rows), np.random.randint(environment_columns)
    return current_row_index, current_column_index

def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < 1 - epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index, new_column_index = current_row_index, current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

# Q-learning parameters
epsilon = 0.1
discount_factor = 0.9
learning_rate = 0.9

# Re-initialize Q-values and reset the cumulative rewards
q_values = np.zeros((environment_rows, environment_columns, 4))
cumulative_rewards = []
episode_rewards = []  # List to store the reward of each episode for plotting

for episode in range(1000):
    row_index, column_index = get_starting_location()
    episode_reward = 0  # Reset the reward for the episode

    # Run the episode
    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(row_index, column_index, action_index)

        reward = rewards[row_index, column_index]
        episode_reward += reward  # Accumulate rewards for the episode

        # Update Q-values using the Q-learning update rule
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        q_values[old_row_index, old_column_index, action_index] = old_q_value + (learning_rate * temporal_difference)

        # If the agent hits a terminal state, break the loop
        if reward == -100:
            break

    # Store the episode reward
    episode_rewards.append(episode_reward)
    # Accumulate the rewards over episodes
    cumulative_rewards.append(sum(episode_rewards))

# Plot the cumulative rewards over episodes
plt.figure(figsize=(12, 6))
plt.plot(cumulative_rewards, label='Cumulative Rewards')
plt.xlabel('Episode')
plt.ylabel('Cumulative Rewards')
plt.title('Cumulative Total Rewards Over Episodes')
plt.legend()
plt.grid(True)
plt.show()

