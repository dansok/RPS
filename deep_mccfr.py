import random

import torch
import torch.nn.functional as F
from torch import optim

from action import Action
from strategy import KAN


class DeepMCCFR:
    def __init__(
            self,
            iterations=1_000_000,
            batch_size=32,
            memory_size=1_000_000,
            lr=0.00001,
    ):
        self.num_actions = len(Action)
        self.iterations = iterations
        self.batch_size = batch_size

        # # Value networks for advantage estimation
        # self.advantage_nets = [
        #     StrategyNet(input_size=1, output_size=self.num_actions),  # Player 1
        #     StrategyNet(input_size=1, output_size=self.num_actions)  # Player 2
        # ]

        # Use KAN networks instead of regular neural nets
        self.advantage_nets = [
            KAN(input_size=1, output_size=self.num_actions, num_inner_funcs=10),
            KAN(input_size=1, output_size=self.num_actions, num_inner_funcs=10)
        ]

        # Optimizers
        self.optimizers = [
            optim.Adam(net.parameters(), lr=lr)
            for net in self.advantage_nets
        ]

        # Replay buffers for advantages
        self.advantage_memories = [
            [],  # Player 1's memory
            []  # Player 2's memory
        ]
        self.memory_size = memory_size

        # Initialize utility matrix
        self.utility_matrix = torch.tensor(
            [[0, -1, 1],  # R vs (R, P, S)
             [1, 0, -1],  # P vs (R, P, S)
             [-1, 1, 0]],  # S vs (R, P, S)
            dtype=torch.float32
        )

    @staticmethod
    def get_strategy(advantages, explore=True, temperature=1.0):
        """Convert advantages to strategy using Boltzmann distribution."""
        if explore:
            # Add exploration noise
            advantages = advantages + torch.randn_like(advantages) * 0.1

        # Apply temperature for Boltzmann exploration
        logits = advantages / temperature
        return F.softmax(logits, dim=-1)

    def sample_opponent_strategy(self, player):
        """Sample opponent strategy using current advantage network."""
        with torch.no_grad():
            x = torch.tensor([[0.0]], dtype=torch.float32)
            advantages = self.advantage_nets[1 - player](x).squeeze()
            strategy = DeepMCCFR.get_strategy(advantages, explore=True)
            return strategy

    def mc_traverse(self, player, num_samples=10):
        """Monte Carlo traversal for collecting advantage samples."""
        advantages_sum = torch.zeros(self.num_actions)

        # Get current player's advantages
        with torch.no_grad():
            x = torch.tensor([[0.0]], dtype=torch.float32)
            current_advantages = self.advantage_nets[player](x).squeeze()
            current_policy = self.get_strategy(current_advantages, explore=True)

        # Monte Carlo sampling
        for _ in range(num_samples):
            # Sample opponent strategy
            opponent_strategy = self.sample_opponent_strategy(player)

            # Sample actions
            action = torch.multinomial(current_policy, 1).item()
            opp_action = torch.multinomial(opponent_strategy, 1).item()

            # Calculate utility
            utility = self.utility_matrix[action, opp_action].item()
            if player == 1:  # Flip utility for player 2
                utility = -utility

            # Calculate counterfactual values for all actions
            cf_values = torch.zeros(self.num_actions)
            for a in range(self.num_actions):
                if player == 0:
                    cf_values[a] = self.utility_matrix[a, opp_action].item()
                else:
                    cf_values[a] = -self.utility_matrix[opp_action, a].item()

            advantages_sum += cf_values - utility

        # Average advantages over samples
        advantages = advantages_sum / num_samples

        # Store experience
        self.advantage_memories[player].append({
            'info_state': x,
            'advantages': advantages,
        })

        # Maintain fixed buffer size
        if len(self.advantage_memories[player]) > self.memory_size:
            self.advantage_memories[player].pop(0)

    def train_advantage_network(self, player):
        """Train advantage network using collected samples."""
        if len(self.advantage_memories[player]) < self.batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(self.advantage_memories[player], self.batch_size)

        # Prepare batch data
        states = torch.cat([x['info_state'] for x in batch])
        advantages = torch.stack([x['advantages'] for x in batch])

        # Train network
        self.optimizers[player].zero_grad()
        predicted_advantages = self.advantage_nets[player](states)
        loss = F.mse_loss(predicted_advantages, advantages)
        loss.backward()
        self.optimizers[player].step()

        return loss.item()

    def train(self):
        """Main training loop."""
        for iteration in range(self.iterations):
            # Monte Carlo sampling for both players
            self.mc_traverse(player=0)
            self.mc_traverse(player=1)

            # Train networks
            loss_1 = self.train_advantage_network(player=0)
            loss_2 = self.train_advantage_network(player=1)

            if iteration % 1000 == 0:
                with torch.no_grad():
                    x = torch.tensor([[0.0]], dtype=torch.float32)
                    advantages_1 = self.advantage_nets[0](x).squeeze()
                    advantages_2 = self.advantage_nets[1](x).squeeze()
                    strategy_1 = self.get_strategy(advantages_1, explore=False)
                    strategy_2 = self.get_strategy(advantages_2, explore=False)

                    print(f"Iteration {iteration}:")
                    print(f"Loss 1: {loss_1:.6f}, Loss 2: {loss_2:.6f}")
                    print(f"Strategy 1: {strategy_1.numpy()}")
                    print(f"Strategy 2: {strategy_2.numpy()}")
                    print("-" * 50)


def main():
    deep_mc_cfr = DeepMCCFR(iterations=50000)
    deep_mc_cfr.train()

    # Get final strategies
    with torch.no_grad():
        x = torch.tensor([[0.0]], dtype=torch.float32)
        advantages_1 = deep_mc_cfr.advantage_nets[0](x).squeeze()
        advantages_2 = deep_mc_cfr.advantage_nets[1](x).squeeze()

        final_strategy_1 = deep_mc_cfr.get_strategy(advantages_1, explore=False)
        final_strategy_2 = deep_mc_cfr.get_strategy(advantages_2, explore=False)

    print("\nFinal Strategies:")
    print("Player 1:", final_strategy_1.numpy())
    print("Player 2:", final_strategy_2.numpy())

    # Evaluate
    print("\nEvaluating strategies...")
    wins_1 = wins_2 = ties = 0
    num_games = 1000

    for _ in range(num_games):
        action_1 = torch.multinomial(final_strategy_1, 1).item()
        action_2 = torch.multinomial(final_strategy_2, 1).item()

        utility = deep_mc_cfr.utility_matrix[action_1, action_2].item()
        if utility > 0:
            wins_1 += 1
        elif utility < 0:
            wins_2 += 1
        else:
            ties += 1

    print(f"\nResults over {num_games} games:")
    print(f"Player 1 wins: {wins_1 / num_games:.1%}")
    print(f"Player 2 wins: {wins_2 / num_games:.1%}")
    print(f"Ties: {ties / num_games:.1%}")


if __name__ == "__main__":
    main()
