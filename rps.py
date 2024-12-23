import random
from types import MappingProxyType

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim

from action import Action
from strategy import StrategyNet


# Utility function for RPS
def get_utility(my_action, opponent_action):
    """Returns the utility for my_action vs opponent_action."""
    if my_action == opponent_action:
        return 0  # Tie
    elif ((my_action == Action.ROCK and opponent_action == Action.SCISSORS) or
          (my_action == Action.SCISSORS and opponent_action == Action.PAPER) or
          (my_action == Action.PAPER and opponent_action == Action.ROCK)):
        return 1  # Win
    else:
        return -1  # Loss


class MCCFR:
    def __init__(self, iterations=1_000_000, epsilon=0.9, epsilon_decay=0.9999, min_epsilon=0.1):
        self.num_actions = len(Action)
        self.iterations = iterations
        self.epsilon = epsilon  # Exploration rate for epsilon-greedy
        self.epsilon_decay = epsilon_decay  # Decay factor for epsilon over time
        self.min_epsilon = min_epsilon  # Minimum epsilon value

        # Neural network for approximating strategies
        self.strategy_net = StrategyNet(input_size=1, output_size=self.num_actions)
        self.optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00001)

    def get_strategy_from_net(self):
        """Retrieve the strategy as a valid probability distribution."""
        x = torch.tensor([[0.0]], dtype=torch.float32)  # Fixed input for state-independent strategy
        strategy = self.strategy_net(x).squeeze(dim=0)  # Extract the output layer
        return strategy

    def sample_action(self, strategy):
        """Sample an action based on the given strategy."""
        return Action(random.choices(range(self.num_actions), weights=strategy.tolist(), k=1)[0])

    def epsilon_greedy_action(self, strategy):
        """Select an action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return Action.get_random()
        else:
            return self.sample_action(strategy)

    @staticmethod
    def compute_sampled_regrets(
            my_action,
            opponent_action_distribution=MappingProxyType({
                Action.ROCK: np.float32(7 / 10),
                Action.PAPER: np.float32(2 / 10),
                Action.SCISSORS: np.float32(1 / 10),
            }),
            num_samples=2,
    ):
        """Compute regrets based on sampled opponent actions."""
        regrets = np.zeros(3)

        # Sample opponent actions
        sampled_opponent_actions = random.choices(
            population=list(opponent_action_distribution.keys()),
            weights=list(opponent_action_distribution.values()),
            k=num_samples,
        )

        # Compute utility differences
        my_utilities = [get_utility(my_action, opp_action) for opp_action in sampled_opponent_actions]
        for action in Action:
            action_utilities = [get_utility(action, opp_action) for opp_action in sampled_opponent_actions]
            regrets[action.value] = np.mean([a - b for a, b in zip(action_utilities, my_utilities)])

        return regrets

    @staticmethod
    def simulate_win_rates(
            opponent_action_distribution=MappingProxyType({
                Action.ROCK: np.float32(7 / 10),
                Action.PAPER: np.float32(2 / 10),
                Action.SCISSORS: np.float32(1 / 10),
            }),
            num_simulations=2,
            num_sampled_actions=2,
    ):
        """
        Simulate win rates for a random subset of actions using MC sampling.

        Args:
            opponent_action_distribution (Mapping[Action, float]): Distribution of opponent's actions.
            num_simulations (int): Number of games to simulate per action.
            num_sampled_actions (int): Number of actions to randomly select for simulation.

        Returns:
            torch.Tensor: A tensor containing win rates for each action, with unknown actions set to 0.
        """
        # Randomly select a subset of actions
        sampled_actions = random.sample(Action.list_actions(), k=num_sampled_actions)
        win_counts = np.zeros(len(Action))  # Initialize win counts for all actions

        # Simulate games only for the sampled actions
        for action in sampled_actions:
            for _ in range(num_simulations):
                # Sample an opponent action
                opponent_action = random.choices(
                    population=list(opponent_action_distribution.keys()),
                    weights=list(opponent_action_distribution.values()),
                )[0]

                # Check if this action wins
                if get_utility(action, opponent_action) > 0:
                    win_counts[action.value] += 1

        # Convert win counts to win rates (only for sampled actions)
        win_rates = win_counts / num_simulations
        return torch.tensor(win_rates, dtype=torch.float32)

    import matplotlib.pyplot as plt

    def train(self):
        """Train the strategy network to minimize KL divergence between strategy and win rates."""
        loss_values = []  # List to store loss values for plotting

        # Set up the real-time plot
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], label="Loss", color="blue")
        ax.set_xlim(0, self.iterations)
        ax.set_ylim(-1, 1)  # Adjust this based on expected loss range
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Iterations")
        ax.legend()
        ax.grid(True)

        for iteration in range(self.iterations):
            # Get strategy from the network (probability distribution)
            network_strategy = self.get_strategy_from_net()  # Already a PyTorch tensor

            # Simulate win rates for all actions using MC sampling
            win_rate_tensor = MCCFR.simulate_win_rates()

            # Compute KL divergence loss
            loss = F.kl_div(network_strategy.log(), win_rate_tensor, reduction='batchmean')

            # Perform optimization step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update epsilon for exploration-exploitation balance
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Log progress and record loss

            if iteration % 1_000 == 0:
                loss_values.append(loss.item())
                print(f"({iteration}) loss = {loss.item():.6f}; strategy = {network_strategy.detach().numpy()}")

                line.set_xdata(range(len(loss_values)))
                line.set_ydata(loss_values)
                ax.set_ylim(0, max(loss_values) * 1.1)  # Dynamically adjust y-axis
                fig.canvas.draw()
                fig.canvas.flush_events()

        plt.ioff()  # Disable interactive mode after training
        plt.show()  # Finalize the plot

        return self.get_strategy_from_net()

    def play(self, rounds=10):
        """Play RPS using the trained strategy."""
        print("\n--- Playing Against MCCFR Agent ---")
        for i in range(rounds):
            strategy = self.get_strategy_from_net()
            agent_action = self.epsilon_greedy_action(strategy)
            opponent_action = Action.get_random()
            print(f"Round {i + 1}: Agent={agent_action}, Opponent={opponent_action}")
            if agent_action == opponent_action:
                print("Result: Tie!")
            elif get_utility(agent_action, opponent_action) > 0:
                print("Result: Agent Wins!")
            else:
                print("Result: Opponent Wins!")


def main():
    mccfr = MCCFR(epsilon=0.2)  # Initialize MCCFR
    initial_strategy = mccfr.get_strategy_from_net().detach().numpy()
    print("Initial Strategy (Untrained):", initial_strategy)
    trained_strategy = mccfr.train().detach().numpy()
    print("Trained Strategy:", trained_strategy)
    mccfr.play(rounds=5)


if __name__ == "__main__":
    main()
