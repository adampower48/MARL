import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.nn.modules.loss import MSELoss
from torch.optim import Adam

from CNN import SimpleCNN
from tic_tac_toe_game import TicTackToeGame


def grid_to_cnn_input(grid, num_classes):
    grid = torch.from_numpy(grid)  # convert to tensor
    encoded = one_hot(grid, num_classes)  # convert to one-hot
    cnn_input = encoded.permute(2, 0, 1).unsqueeze(0)  # (w, h, c) -> (n, c, w, h)

    return cnn_input.float()


def cnn_output_to_grid(output):
    # Convert to numpy & remove batch dim
    return output[0].detach().numpy()


def get_action(game: TicTackToeGame, player_model, player_idx, random_chance=0):
    if np.random.rand() < random_chance:
        # Random action
        return np.zeros(game.get_action_space()[1][1]), np.random.choice(game.get_valid_actions(player_idx))

    # Get game state
    grid = game.get_state()
    state = grid_to_cnn_input(grid + 1, NUM_PLAYERS + 1)

    # Raw predictions
    with torch.no_grad():
        preds = player_model(state)
    preds = cnn_output_to_grid(preds)

    # Convert to action
    legal_actions = game.get_valid_actions(player_idx)

    valid_actions = np.arange(GRID_SIZE * GRID_SIZE)[legal_actions]
    valid_preds = preds.ravel()[legal_actions]

    max_action = valid_actions[np.argmax(valid_preds)]

    return preds, max_action


def train(player_model, player_idx, states, actions, is_winner):
    # Get rewards for each state
    rewards = [REWARD_SIZE * (1 - EPSILON_DECAY) ** i for i in range(len(states))][::-1]
    if not is_winner:
        rewards = [-r for r in rewards]

    # print("Rewards:", rewards)

    # Get model predictions
    cnn_inputs = [grid_to_cnn_input(s + 1, NUM_PLAYERS + 1) for s in states]
    cnn_inputs = torch.cat(cnn_inputs)
    cnn_outputs = player_model(cnn_inputs)

    # Compute pseudo-ground truth
    cnn_gt = []
    out = cnn_outputs.detach().numpy()
    for s, r, a, y in zip(states, rewards, actions, out):
        gt = np.where(s == -1, y, 0)  # empty space
        gt.ravel()[a] = y.ravel()[a] + r  # taken action
        gt = gt.clip(0, 1, out=gt)
        cnn_gt.append(gt)

    # Create tensors
    cnn_gt = torch.tensor(cnn_gt)

    # Compute loss
    loss = MSELoss()(cnn_outputs, cnn_gt)
    print(loss)

    # Backprop gradients
    player_model.zero_grad()
    loss.backward()

    # Update weights
    optimizer = Adam(player_model.parameters(), lr=LEARNING_RATE)
    optimizer.step()


def main():
    # Set up game & players
    game = TicTackToeGame(GRID_SIZE, NUM_PLAYERS)
    players = [SimpleCNN(NUM_PLAYERS + 1, filter_size=CNN_FILTER_SIZE) for _ in range(NUM_PLAYERS)]
    # players = [RandomCNN(), SimpleCNN(NUM_PLAYERS + 1, filter_size=CNN_FILTER_SIZE)]

    winners = []
    for i in range(NUM_GAMES):
        game_states = []
        predicted_actions = []
        game.reset()

        # Play game
        cur_round = 0
        cur_player = 0
        while not game.is_game_over() and (winner := game.get_winner()) is None:
            # print("Round:", cur_round)
            # print("Player:", cur_player)

            # Get & perform action
            game_states.append(game.get_state().copy())
            preds, action = get_action(game, players[cur_player], cur_player)
            game.perform_action(cur_player, action)
            predicted_actions.append(action)

            # print("Action:", action)
            # print(game.get_state())
            # print(preds)

            cur_player = (cur_player + 1) % NUM_PLAYERS
            cur_round += 1

        print("Game:", i + 1)
        print(game.get_state())

        with open("winners.txt", "a") as f:
            print("X" if winner is None else winner, end="", file=f)

        # Train
        print("Winner:", winner)
        winners.append(winner)
        for agent_idx in range(NUM_PLAYERS):
            states = game_states[agent_idx::NUM_PLAYERS]
            actions = predicted_actions[agent_idx::NUM_PLAYERS]

            if players[agent_idx].can_train:
                train(players[agent_idx], agent_idx, states, actions, agent_idx == winner)

    # Print game stats
    print("Stats:")
    for i in range(NUM_PLAYERS):
        wins = len([w for w in winners if w == i])
        print(f"Player {i}: {wins / len(winners):.1%}")

    draws = len([w for w in winners if w is None])
    print(f"Draws    : {draws / len(winners):.1%}")


if __name__ == '__main__':
    GRID_SIZE = 3
    NUM_PLAYERS = 2
    EPSILON_DECAY = 0.1
    LEARNING_RATE = 0.01
    REWARD_SIZE = 0.1
    RANDOM_CHANCE = 0.05
    NUM_GAMES = 10000
    CNN_FILTER_SIZE = 5

    main()
