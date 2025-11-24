from random import random
from sys import argv

import ale_py
import torch
from gymnasium.wrappers import RecordVideo

from dqn import DQN, device, evaluate, make_environment

checkpoint_path = argv[1]

environment = make_environment(is_rendering=True)

returns = evaluate(environment, lambda _: torch.tensor([[random() for _ in range(9)]]))

print("Random returns:", returns)
print("Average random return:", sum(returns) / len(returns))

environment = RecordVideo(
    environment,
    video_folder="./videos",
    episode_trigger=lambda _: True,
)

model = DQN().to(device).eval()
model.load_state_dict(torch.load(checkpoint_path))

returns = evaluate(environment, model)

print("Model returns:", returns)
print("Average model return:", sum(returns) / len(returns))
