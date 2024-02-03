This project implemented a Q-learning algorithm in Python, aiming to guide a warehouse robot through a grid-simplified environment, focusing on avoiding shelves and efficiently reaching the shipping area.

## MDP Problem Formulation
### Goal and Assumptions
The goal of the problem is to train the robot to learn the optimal policy and find the shortest path within the warehouse using the Q-learning algorithm and maximize total rewards.
The main assumption is that the warehouse environment is deterministic and constant. For example, the layout remains unchanged through the training process, including the composition of the grid, the location of the shelves, the location of the shipping area, and the area that the robot can navigate.


### Agent/Environment
Agent: A simulated warehouse robot.

Environment: The environment is a grid-based simulated warehouse. The squares in the grid include the area that robot can navigate, shelves (obstacles) and the shipping area (destination). The simulated layout of this problem provides a real environment for the robot to navigate and complete the task.

<img width="346" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/abdb9ce3-0223-4a62-904e-d041ea2663f6">


### Action Spaces
The robot can move in four directions: Up, Right, Down, and Left, to navigate through the warehouse.

### Transition Dynamics
Transitions depend on the robot's actions, moving to new locations, avoiding obstacles, or reaching the destination.

### Reward Definition
<img width="324" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/8d336d69-342d-4b39-9b4d-47340b9cf1a3">

Rewards are assigned to encourage efficient navigation: -1 for each move, -100 for collision, and +100 for reaching the shipping area.

### MDP Problem Classification
This problem is episodic, deterministic, and involves a finite state space, making it suitable for Q-learning.

### Solution Methods
Q-learning is chosen for its flexibility, convergence guarantees, and ability to operate without a model of the environment. It's a balance between exploration and exploitation, adapting to various scenarios within the warehouse.

### Q-learning Workflow Diagram
<img width="262" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/6af7fb2b-7f22-404c-b8c1-a98e80ca8686">

### General Flow of Implementation
#### Initialization
The environment is set up as an 11x11 grid where each square and associated action (up, right, down, left) is assigned a zero-initialized Q-value matrix. 
Initially, the rewards grid is configured with all values set to -100. Specific locations in the grid are assigned specific rewards: for example, the shipping area is set to 100, while shelves are set to -100. Aisle locations, which represent navigable paths for the robot, are defined with a reward value of -1, indicating permissible movement areas. 
#### Defining Essential Utility Functions
check_terminal_state: Checks if a given state is a terminal state.
random_start: Randomly selects a starting location that is not a terminal state.
select_action: Implements the epsilon-greedy strategy for selecting next action.
calculate_next_position: Determines the next location based on the current location and action.
find_shortest_path: Finds the shortest path from a given starting location to the shipping area.
#### Training the Agent Using Q-Learning
In the process of training the agent using the Q-learning algorithm, key parameters are defined as follows: an epsilon value of 0.1 for the exploration-exploitation trade-off, a discount factor of 0.9 that weighs future rewards, and a learning rate of 0.9 to control the speed of updates in Q-values.
The training process includes the following steps:
1.	Randomly choose a non-terminal state as the starting location or find the start location according to the given start column and row indexes for each new episode.

2.	Choose an action (up, right, down, or left) for the current state based on the epsilon-greedy algorithm. The value of ε is set to 0.1 in the project, which means that the algorithm will have a probability of 90% to choose the known action that can maximize the agent’s reward each time, and a probability of 10% to choose a random action to encourage the agent to explore the environment. Typically the smaller the value of ε, the higher the long term returns will have.

3.	Perform the chosen action and move to the next location. 

4.	Upon arriving at the new location, the agent receives the corresponding reward and evaluate the estimated maximum future reward for that state, followed by the calculation of the temporal difference value.

5.	Update the Q-value for the previous state based on the previous Q-value and the value of temporal difference.

6.	If the new state is a terminal state, then begin a new episode. Otherwise, go to step 2.
This entire process will be repeated 1000 episodes. This will provide enough iterations for the agent to learn the most efficient paths between all the white squares in the warehouse and the shipping area, while simultaneously avoiding crashing into any shelves and maximizing the total rewards.
#### Visualizing Results
Optimal Path Result：

<img width="506" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/1cf1fe9f-f260-40e5-bb18-260bab79ddc3">

Optimal Path Learned from Starting Location (9, 7) to Shipping Area (0, 5) , Plotted by Python：

<img width="403" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/f9804c1a-977a-485b-9051-004e71432cb1">

Reverse Optimal Path Result：

<img width="506" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/db58b792-5362-496f-a7e3-6a1924844e3f">

Reverse Optimal Path Learned from Shipping Area (0, 5) to Location (5, 2) , Plotted by Python：

<img width="392" alt="image" src="https://github.com/mylittleZ/Warehouse-Robot-with-Q-learning/assets/30174451/911f19eb-7379-4325-8dc0-f41c0dcb50fd">







