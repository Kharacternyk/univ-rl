from collections import deque
from dataclasses import dataclass
from itertools import count
from random import randint, random, sample

import ale_py
import gymnasium
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    next_state: torch.Tensor | None
    reward: float


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_convolution = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=16, stride=8
        )
        self.second_convolution = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2
        )
        self.first_linear = nn.Linear(288, 128)
        self.second_linear = nn.Linear(128, 9)
        self.activation = nn.Sequential(
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.activation(self.first_convolution(x))
        x = self.activation(self.second_convolution(x))

        x = x.view(x.size(0), -1)
        x = self.activation(self.first_linear(x))
        x = self.second_linear(x)

        return x


def state_to_torch(state):
    return torch.from_numpy(state).to(device)


def make_environment(is_rendering=False):
    render_mode = "rgb_array" if is_rendering else None
    environment = gymnasium.make(
        "ALE/LostLuggage-v5", render_mode=render_mode, frameskip=1
    )
    environment = gymnasium.wrappers.AtariPreprocessing(
        environment,
        noop_max=0,
        scale_obs=True,
        terminal_on_life_loss=False,
        grayscale_obs=True,
    )
    environment = gymnasium.wrappers.FrameStackObservation(environment, 4)
    return environment


@torch.inference_mode()
def evaluate(environment, model, n=10):
    returns = []

    for _ in range(n):
        state = state_to_torch(environment.reset()[0]).unsqueeze(0)
        episode_return = 0

        while True:
            values = model(state)
            action = values.max(1).indices.item()

            next_state, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward

            if terminated or truncated:
                break

            state = state_to_torch(next_state).unsqueeze(0)

        environment.close()
        returns.append(episode_return)

    return returns


if __name__ == "__main__":
    policy_network = DQN().to(device).train()
    target_network = DQN().to(device).eval()
    policy_network.load_state_dict(target_network.state_dict())

    replay_buffer = deque(maxlen=150_000)
    optimizer = torch.optim.AdamW(policy_network.parameters(), lr=3e-4, amsgrad=True)

    environment = make_environment()
    criterion = nn.SmoothL1Loss()
    returns = deque(maxlen=20)
    epsilon = 0.2

    for episode_index in range(1000):
        state = state_to_torch(environment.reset()[0]).unsqueeze(0)
        episode_return = 0

        for step_index in count():
            if random() < epsilon:
                action = randint(0, 8)
            else:
                with torch.inference_mode():
                    action = policy_network(state).max(1).indices.item()

            next_state, reward, terminated, truncated, _ = environment.step(action)
            episode_return += reward

            if terminated:
                next_state = None
            else:
                next_state = state_to_torch(next_state).unsqueeze(0)

            replay_buffer.append(Transition(state, action, next_state, reward))

            state = next_state
            batch_size = 512

            if not ((step_index + 1) % 25) and len(replay_buffer) >= batch_size * 2:
                transitions = sample(replay_buffer, batch_size)

                states = torch.cat([transition.state for transition in transitions])
                actions = torch.tensor(
                    [transition.action for transition in transitions], device=device
                )
                rewards = torch.tensor(
                    [transition.reward for transition in transitions], device=device
                )

                values = policy_network(states).gather(1, actions.unsqueeze(1))

                next_values = torch.zeros(batch_size, device=device)

                with torch.inference_mode():
                    next_values[
                        [
                            i
                            for i, transition in enumerate(transitions)
                            if transition.next_state is not None
                        ]
                    ] = (
                        target_network(
                            torch.cat(
                                [
                                    transitions.next_state
                                    for transitions in transitions
                                    if transitions.next_state is not None
                                ]
                            )
                        )
                        .max(1)
                        .values
                    )

                expected_values = next_values * 0.9995 + rewards
                loss = criterion(values, expected_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                target_network_dict = target_network.state_dict()
                policy_network_dict = policy_network.state_dict()

                for key in policy_network_dict:
                    tau = 5e-3
                    target_network_dict[key] = policy_network_dict[
                        key
                    ] * tau + target_network_dict[key] * (1 - tau)

            if terminated or truncated:
                break

        returns.append(episode_return)

        if not ((episode_index + 1) % 5):
            test_returns = evaluate(environment, policy_network, 4)

            print(
                f"Episode {episode_index + 1}, "
                f"rolling avg train return: {sum(returns) / len(returns):.2f}, "
                f"avg test return: {sum(test_returns) / len(test_returns):.2f}, "
                f"epsilon: {epsilon:.2f}"
            )

            torch.save(
                policy_network.state_dict(), f"checkpoint_{episode_index + 1}.pt"
            )
