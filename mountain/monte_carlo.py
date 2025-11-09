import random
from dataclasses import dataclass
from math import cos

import torch
from gymnasium import make

actions = range(3)


def discretize_position(position, position_discrete_count):
    return round((1.2 + position) / 1.8 * position_discrete_count)


def discretize_velocity(velocity, velocity_discrete_count):
    return round((0.07 + velocity) / 0.14 * velocity_discrete_count)


def get_epsilon_greedy_action(
    value_table, discrete_position, discrete_velocity, epsilon
):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)

    best_action = None
    best_value = None

    for action in actions:
        value = value_table[discrete_position, discrete_velocity, action]

        if best_value is None or value > best_value:
            best_action = action
            best_value = value

    assert best_action is not None
    return best_action


@dataclass
class TimeStep:
    discrete_position: int
    discrete_velocity: int
    action: int
    training_reward: float
    testing_reward: float
    is_first_visit: bool


def get_episode(
    environment,
    value_table,
    epsilon,
    max_time,
):
    episode = []
    position, velocity = environment.reset()[0]
    visited_states = set()

    for _ in range(max_time):
        discrete_position = discretize_position(position, value_table.size(0))
        discrete_velocity = discretize_velocity(velocity, value_table.size(1))
        action = get_epsilon_greedy_action(
            value_table, discrete_position, discrete_velocity, epsilon
        )

        state_action = (
            discrete_position,
            discrete_velocity,
            action,
        )

        is_first_visit = state_action in visited_states
        visited_states.add(state_action)

        (new_position, new_velocity), testing_reward, terminated, _truncated, *_ = (
            environment.step(action)
        )

        new_energy = (new_velocity**2 / 0.005) + abs(cos(3 * new_position))

        training_reward = new_energy / 1.5 - 1

        episode.append(
            TimeStep(*state_action, training_reward, testing_reward, is_first_visit)
        )

        if terminated:
            break

        position = new_position
        velocity = new_velocity

    return episode


def get_greedy_policy_table(value_table):
    policy_table = torch.empty(value_table.shape[:-1])

    for discrete_position in range(value_table.size(0)):
        for discrete_velocity in range(value_table.size(1)):
            best_action = None
            best_value = None

            for action in actions:
                value = value_table[discrete_position, discrete_velocity, action]

                if best_value is None or value > best_value:
                    best_action = action
                    best_value = value

            assert best_action is not None
            policy_table[discrete_position, discrete_velocity] = best_action

    return policy_table


if __name__ == "__main__":
    environment = make("MountainCar-v0")

    value_table = torch.zeros(20, 20, 3)
    total_returns = torch.zeros_like(value_table)
    visit_counts = torch.zeros_like(value_table)

    episode_count = 10000
    epsilon = 0.22
    epsilon_decay_factor = 0.9999

    for i in range(episode_count):
        episode = get_episode(environment, value_table, epsilon, 5000)
        total_training_return = sum(step.training_reward for step in episode)

        if not i % 100:
            total_testing_return = sum(step.testing_reward for step in episode)
            print(
                f"Episode {i + 1}, "
                f"training return: {total_training_return:.1f}, "
                f"testing return: {total_testing_return:.1f}, "
                f"epsilon: {epsilon:.2f}"
            )

        epsilon *= epsilon_decay_factor

        for step in episode:
            if not step.is_first_visit:
                continue

            state_action = (step.discrete_position, step.discrete_velocity, step.action)

            total_returns[state_action] += total_training_return
            visit_counts[state_action] += 1
            value_table[state_action] = (
                total_returns[state_action] / visit_counts[state_action]
            )

            total_training_return -= step.training_reward

    optimal_policy_table = get_greedy_policy_table(value_table)

    print()
    print("Optimal policy table:")
    print(
        "\n".join(
            "".join(str(int(element)) for element in row)
            for row in optimal_policy_table
        ),
    )
    print()

    episode_count = 100
    total_return = 0

    for _ in range(episode_count):
        episode = get_episode(environment, value_table, 0, 200)
        total_return += sum(step.testing_reward for step in episode)

    print("Average return:", total_return / episode_count)
