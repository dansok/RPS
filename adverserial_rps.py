import random

import torch
import torch.nn.functional as F
from torch import optim

from action import Action
from strategy import StrategyNet


class MCCFR:
    def __init__(
            self,
            iterations=1_000_000,
            epsilon=0.9,
            epsilon_decay=0.9999,
            min_epsilon=0.1,
            entropy_weight=0.1,
    ):
        self.num_actions = len(Action)
        self.iterations = iterations
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.entropy_weight = entropy_weight  # Weight for entropy regularization

        # Initialize networks with separate computation graphs
        self.strategy_net = StrategyNet(input_size=1, output_size=self.num_actions)
        self.adversary_net = StrategyNet(input_size=1, output_size=self.num_actions)

        # Use separate optimizers with smaller learning rates
        self.optimizer_1 = optim.Adam(self.strategy_net.parameters(), lr=0.0001)
        self.optimizer_2 = optim.Adam(self.adversary_net.parameters(), lr=0.0001)

        # Initialize the utility matrix once
        self.utility_matrix = torch.tensor(
            [[0, -1, 1],  # R vs (R, P, S)
             [1, 0, -1],  # P vs (R, P, S)
             [-1, 1, 0]],  # S vs (R, P, S)
            dtype=torch.float32,
        )

    @staticmethod
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

    @staticmethod
    def entropy_loss(probs):
        """Compute entropy of probability distribution."""
        return -(probs * torch.log(probs + 1e-10)).sum()

    @staticmethod
    def get_strategy_from_net(net):
        """Get strategy distribution from network with detached computation."""
        with torch.no_grad():
            x = torch.tensor([[0.0]], dtype=torch.float32)
            logits = net(x).squeeze(dim=0)
            return F.softmax(logits, dim=0)

    @staticmethod
    def get_strategy_for_training(net):
        """Get strategy distribution from network for training."""
        x = torch.tensor([[0.0]], dtype=torch.float32)
        logits = net(x).squeeze(dim=0)
        return F.softmax(logits, dim=0)

    def train(self):
        """Train both networks with entropy regularization."""
        for iteration in range(self.iterations):
            # Forward pass for player 1
            strategy_1 = MCCFR.get_strategy_for_training(self.strategy_net)

            with torch.no_grad():
                strategy_2 = MCCFR.get_strategy_from_net(self.adversary_net)

            # Compute utility and entropy regularization for player 1
            expected_utility_1 = torch.matmul(
                strategy_1,
                torch.matmul(self.utility_matrix, strategy_2)
            )
            entropy_term_1 = MCCFR.entropy_loss(strategy_1)

            # Loss with entropy regularization
            loss_1 = -expected_utility_1 - self.entropy_weight * entropy_term_1
            # loss_1 = -expected_utility_1

            # Update player 1
            self.optimizer_1.zero_grad()
            loss_1.backward()
            self.optimizer_1.step()

            # Forward pass for player 2
            with torch.no_grad():
                strategy_1 = MCCFR.get_strategy_from_net(self.strategy_net)
            strategy_2 = MCCFR.get_strategy_for_training(self.adversary_net)

            # Compute utility and entropy regularization for player 2
            expected_utility_2 = torch.matmul(
                strategy_2,
                torch.matmul(self.utility_matrix, strategy_1)
            )
            entropy_term_2 = MCCFR.entropy_loss(strategy_2)

            # Loss with entropy regularization
            loss_2 = -expected_utility_2 - self.entropy_weight * entropy_term_2
            # loss_2 = -expected_utility_2

            # Update player 2
            self.optimizer_2.zero_grad()
            loss_2.backward()
            self.optimizer_2.step()

            # Update epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            # Log progress
            if iteration % 1000 == 0:
                with torch.no_grad():
                    kl_from_uniform = self.compute_kl_from_uniform(strategy_1, strategy_2)
                    print(f"Iteration {iteration}:")
                    print(f"Loss 1: {loss_1.item():.6f}, Loss 2: {loss_2.item():.6f}")
                    print(f"Strategy 1: {strategy_1.numpy()}")
                    print(f"Strategy 2: {strategy_2.numpy()}")
                    print(f"Entropy 1: {entropy_term_1.item():.6f}, Entropy 2: {entropy_term_2.item():.6f}")
                    print(f"KL from uniform: {kl_from_uniform:.6f}")
                    print("-" * 50)

        return self.get_strategy_from_net(self.strategy_net), self.get_strategy_from_net(self.adversary_net)

    def compute_kl_from_uniform(self, strategy_1, strategy_2):
        """Compute KL divergence from uniform for monitoring purposes only."""
        uniform = torch.ones(self.num_actions) / self.num_actions
        kl_1 = F.kl_div(torch.log(strategy_1 + 1e-10), uniform, reduction='sum')
        kl_2 = F.kl_div(torch.log(strategy_2 + 1e-10), uniform, reduction='sum')
        return (kl_1 + kl_2) / 2

    def play(self, rounds=10):
        """Play RPS using the trained strategies."""
        print("\n--- Playing Against MCCFR Agent ---")
        strategy1 = self.get_strategy_from_net(self.strategy_net)
        strategy2 = self.get_strategy_from_net(self.adversary_net)

        p1_wins = p2_wins = ties = 0

        for i in range(rounds):
            agent_action = Action(random.choices(range(self.num_actions),
                                                 weights=strategy1.tolist(), k=1)[0])
            opponent_action = Action(random.choices(range(self.num_actions),
                                                    weights=strategy2.tolist(), k=1)[0])

            print(f"Round {i + 1}: Agent={agent_action.name}, Opponent={opponent_action.name}")

            utility = MCCFR.get_utility(agent_action, opponent_action)
            if utility > 0:
                print("Result: Agent Wins!")
                p1_wins += 1
            elif utility < 0:
                print("Result: Opponent Wins!")
                p2_wins += 1
            else:
                print("Result: Tie!")
                ties += 1

        print("\nFinal Results:")
        print(f"Agent Wins: {p1_wins}")
        print(f"Opponent Wins: {p2_wins}")
        print(f"Ties: {ties}")


def main():
    mccfr = MCCFR(epsilon=0.2)
    player1_strategy, player2_strategy = mccfr.train()

    print("\nFinal Strategies:")
    print("Player 1:", player1_strategy.numpy())
    print("Player 2:", player2_strategy.numpy())

    mccfr.play(rounds=10)


if __name__ == "__main__":
    main()
