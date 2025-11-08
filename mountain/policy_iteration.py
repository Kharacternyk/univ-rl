from math import cos
from time import time

import torch
from gymnasium import make


def discretize_position(position, position_discrete_count):
    return round((1.2 + position) / 1.8 * position_discrete_count)


def discretize_velocity(velocity, velocity_discrete_count):
    return round((0.07 + velocity) / 0.14 * velocity_discrete_count)


def get_transition(
    discrete_position,
    discrete_velocity,
    position_discrete_count,
    velocity_discrete_count,
    action,
):
    position = discrete_position / position_discrete_count * 1.8 - 1.2
    velocity = discrete_velocity / velocity_discrete_count * 0.14 - 0.07

    new_velocity = velocity + (action - 1) * 0.001 - cos(3 * position) * 0.0025
    new_position = position + new_velocity

    discrete_new_position = discretize_position(new_position, position_discrete_count)

    if discrete_new_position < 0:
        discrete_new_position = 0
        new_velocity = 0
        new_position = -1.2
    elif discrete_new_position >= position_discrete_count:
        discrete_new_position = position_discrete_count - 1
        new_velocity = 0
        new_position = 0.6

    reward = abs(new_position - position) - 1

    discrete_new_velocity = discretize_velocity(new_velocity, velocity_discrete_count)

    if discrete_new_velocity < 0:
        discrete_new_velocity = 0
    elif discrete_new_velocity >= velocity_discrete_count:
        discrete_new_velocity = velocity_discrete_count - 1

    return discrete_new_position, discrete_new_velocity, reward


def get_value_table(
    policy_table: torch.Tensor,
    threshold,
    max_iteration_count,
    discount_factor,
):
    value_table = torch.zeros_like(policy_table)
    position_discrete_count = policy_table.size(0)
    velocity_discrete_count = policy_table.size(1)

    for _ in range(max_iteration_count):
        new_value_table = torch.empty_like(value_table)

        for discrete_position in range(position_discrete_count):
            for discrete_velocity in range(velocity_discrete_count):
                action = policy_table[discrete_position, discrete_velocity].item()
                discrete_new_position, discrete_new_velocity, reward = get_transition(
                    discrete_position,
                    discrete_velocity,
                    position_discrete_count,
                    velocity_discrete_count,
                    action,
                )

                new_value_table[discrete_position, discrete_velocity] = (
                    reward
                    + discount_factor
                    * value_table[discrete_new_position, discrete_new_velocity]
                )

        tolerance = (new_value_table - value_table).abs().max().item()

        if tolerance <= threshold:
            return new_value_table

        value_table = new_value_table

    return value_table


def get_greedy_policy(value_table: torch.Tensor, discount_factor):
    position_discrete_count = value_table.size(0)
    velocity_discrete_count = value_table.size(1)
    policy_table = torch.empty_like(value_table)

    for discrete_position in range(position_discrete_count):
        for discrete_velocity in range(velocity_discrete_count):
            best_action = None
            best_value = None

            for action in range(3):
                discrete_new_position, discrete_new_velocity, reward = get_transition(
                    discrete_position,
                    discrete_velocity,
                    position_discrete_count,
                    velocity_discrete_count,
                    action,
                )
                value = (
                    reward
                    + discount_factor
                    * value_table[discrete_new_position, discrete_new_velocity]
                )

                if best_value is None or value > best_value:
                    best_action = action
                    best_value = value

            assert best_action is not None

            policy_table[discrete_position, discrete_velocity] = best_action

    return policy_table


def do_policy_iteration(
    max_iteration_count,
    threshold,
    discount_factor,
    position_discrete_count,
    velocity_discrete_count,
):
    policy_table = torch.ones(position_discrete_count, velocity_discrete_count)

    for _ in range(max_iteration_count):
        value_table = get_value_table(
            policy_table, threshold, max_iteration_count, discount_factor
        )
        new_policy_table = get_greedy_policy(value_table, discount_factor)

        if torch.all(new_policy_table == policy_table):
            break

        policy_table = new_policy_table

    return policy_table


if __name__ == "__main__":
    position_discrete_count = 22
    velocity_discrete_count = 22
    start = time()
    optimal_policy_table = do_policy_iteration(
        100,
        0.01,
        0.72,
        position_discrete_count,
        velocity_discrete_count,
    )
    print(f"Converged in {time() - start:.2f} seconds")
    print()
    print("Optimal policy table:")
    print(
        "\n".join(
            "".join(str(int(element)) for element in row)
            for row in optimal_policy_table
        ),
    )
    print()

    env = make("MountainCar-v0")
    total_reward = 0
    episode_count = 5000

    for _ in range(episode_count):
        state = env.reset()[0]

        for _ in range(200):
            position, velocity = state
            discrete_position = discretize_position(position, position_discrete_count)
            discrete_velocity = discretize_velocity(velocity, velocity_discrete_count)

            action = int(optimal_policy_table[discrete_position, discrete_velocity])
            state, reward, done, *_ = env.step(action)

            total_reward += reward

            if done:
                break

    print("Average reward:", total_reward / episode_count)
