import os
import random
import re
from collections import deque

import numpy as np

from av_game import game_space
from misc import get_state_params, get_model, get_state, train_after_episode, sac_models, one_hot, load_json, write_json


def get_trained_models(models_folder):
    # Get indices of all currently saved models

    os.makedirs(models_folder, exist_ok=True)

    model_nums = []
    for filename in os.listdir(models_folder):
        match = re.match(r"(\d+)\.pt$", filename)
        if match:
            model_nums.append(int(match.group(1)))

    return model_nums


def load_models(gs, models_folder, model_type):
    # Load existing models or create new ones

    trained_model_nums = get_trained_models(models_folder)

    models = []
    for i in range(gs.num_agents):
        # Create blank model
        model = get_model(model_type, gs)

        # Load existing model
        if i in trained_model_nums:
            if model.load_model(models_folder, i):
                model.update_target()

        # Random sample from existing models
        elif trained_model_nums:
            idx = random.choice(trained_model_nums)

            if model.load_model(models_folder, idx):
                model.update_target()

        models.append(model)

    return models


def print_state(model_type, iteration, episode, episode_rewards, gs: game_space):
    msg = "\n".join([
        f"Model: {model_type}",
        f"Episode: {episode}",
        f"Iter: {iteration}",
        f"Reward: {episode_rewards:.3f}",
        f"Team 1:",
        f"\tReinforcements: {gs.reinforcements[0]}",
        f"\tLieutenants: {gs.num_lieutenants[0]}",
        f"\tCommander HP: {gs.commanders[0][3]}",
        f"Team 2:",
        f"\tReinforcements: {gs.reinforcements[1]}",
        f"\tLieutenants: {gs.num_lieutenants[1]}",
        f"\tCommander HP: {gs.commanders[1][3]}",
    ])

    print(msg)


def setup_game():
    flatten_state, prev_states = get_state_params(MODEL_TYPE)
    gs = game_space(visible=UNIT_VISIBILITY, prev_states=prev_states,
                    flatten_state=flatten_state, split_layers=True)

    models = load_models(gs, MODELS_FOLDER, MODEL_TYPE)

    # Reload training data

    if os.path.exists(rewards_file := os.path.join(MODELS_FOLDER, "rewards.json")):
        tracked_rewards = load_json(rewards_file)
    else:
        tracked_rewards = []

    if os.path.exists(durations_file := os.path.join(MODELS_FOLDER, "durations.json")):
        tracked_durations = load_json(durations_file)
    else:
        tracked_durations = []

    total_rewards = sum(tracked_rewards)
    episode = len(tracked_rewards)

    # Predictions
    pred_list = [deque() for i in range(gs.num_agents)]
    last_pred = np.zeros((gs.num_agents, gs.num_actions))
    stochastic = list(range(gs.num_agents))

    return gs, models, tracked_rewards, tracked_durations, total_rewards, episode, pred_list, last_pred, stochastic


def single_episode():
    # Play episode
    print("Play")
    num_iters, episode_reward, done = play()

    # Update policies
    print("Train")
    update_policies(models, gs)
    gs.reset()

    # Save episode data
    tracked_rewards.append(episode_reward)
    tracked_durations.append(num_iters)

    write_json(tracked_rewards, os.path.join(MODELS_FOLDER, "rewards.json"))
    write_json(tracked_durations, os.path.join(MODELS_FOLDER, "durations.json"))


def main_loop(episode=0):
    done = False

    while not done:
        # Single play/train
        single_episode()
        print("=" * 100)
        episode += 1

        # Check if done
        if episode >= MAX_EPISODES:
            done = True


def play():
    cur_iters = 0
    episode_reward = 0

    done = False
    while not done:
        cur_iters += 1
        print("\rIter:", cur_iters, end="", flush=True)

        # alive = list(range(gs.num_agents))
        # preds = np.zeros((gs.num_agents, gs.num_actions))
        states_before = []
        states_after = []
        actions = []
        rewards = []

        gs.reset_markers()
        for agent_idx in range(gs.num_agents):
            # x, y, t, u, h = gs.agents[agent_idx]
            agent_state = get_state(gs, agent_idx, MODEL_TYPE)
            states_before.append(agent_state)

            # Get predicted action
            pred = models[agent_idx].get_action(agent_state)
            if MODEL_TYPE in train_after_episode:
                pred = one_hot(pred, gs.num_actions, float)

            if agent_idx in stochastic:
                action = random.randint(0, gs.num_actions - 1)
            else:
                action = np.argmax(pred)

            # Perform action & get reward
            reward = gs.move_agent(agent_idx, action)
            agent_state = get_state(gs, agent_idx, MODEL_TYPE)
            states_after.append(agent_state)

            episode_reward += reward
            rewards.append(reward)

            if MODEL_TYPE in sac_models:
                actions.append(pred)
            else:
                actions.append(action)

            gs.update_agent_positions()

        # Check stop conditions
        winner = gs.get_winner()
        if winner is not None:
            done = True

            # Add winning reward
            for agent_idx in range(gs.num_agents):
                x, y, t, u, h = gs.agents[agent_idx]

                if t == winner:
                    rewards[agent_idx - gs.num_agents] += 1

        elif cur_iters >= MAX_ITERS:
            done = True

        if PRINT_VISUALS and cur_iters % 100 == 0:
            os.system("clear")
            print(gs.print_game_space())
            print()
            print_state(MODEL_TYPE, cur_iters, episode, episode_reward, gs)

        # Train
        train_after_iter(models, gs, states_before, states_after, actions, rewards, done)

    print()
    print_state(MODEL_TYPE, cur_iters, episode, episode_reward, gs)

    return cur_iters, episode_reward, done


def train_after_iter(models, gs, states_before, states_after, actions, rewards, done):
    for agent_idx in range(gs.num_agents):
        models[agent_idx].push_replay(states_before[agent_idx], actions[agent_idx], rewards[agent_idx], done,
                                      states_after[agent_idx])

    for agent_idx in range(gs.num_agents):
        models[agent_idx].train()


def update_policies(models, gs):
    global stochastic
    # Update epsilon
    epsilon = max(EPSILON_MIN, 1 - (total_rewards * EPSILON_DECAY))

    # Train
    for agent_idx in range(gs.num_agents):
        if MODEL_TYPE in train_after_episode:
            print(f"\rTraining model: {agent_idx}", end="", flush=True)
            models[agent_idx].update_policy()

        if (ns := int(epsilon * gs.num_agents)) > 0:
            # stochastic = np.random.choice(range(gs.num_agents), ns, replace=True).tolist()
            stochastic = random.sample(range(gs.num_agents), ns)
        else:
            stochastic = []

        # Save model
        models[agent_idx].save_model(MODELS_FOLDER, agent_idx)

    print()


if __name__ == '__main__':
    # CONSTANTS
    UNIT_VISIBILITY = 4
    MAX_EPISODES = 500
    MAX_ITERS = 100
    EPSILON_DECAY = 0.00001
    EPSILON_MIN = 0
    MODEL_TYPE = "PG"
    # preds_len = 30
    PRINT_VISUALS = False
    PRINT_PREDS = False
    MODELS_FOLDER = f"{MODEL_TYPE}_av_game_save"

    # Setup
    gs, models, tracked_rewards, tracked_durations, total_rewards, episode, pred_list, last_pred, stochastic = setup_game()

    # Main loop
    main_loop(episode)
