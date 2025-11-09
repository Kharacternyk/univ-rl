from gymnasium import make
from gymnasium.wrappers import RecordVideo


def discretize_position(position, position_discrete_count):
    return round((1.2 + position) / 1.8 * position_discrete_count)


def discretize_velocity(velocity, velocity_discrete_count):
    return round((0.07 + velocity) / 0.14 * velocity_discrete_count)


print("Policy table:")
policy_table = []

while row := input():
    policy_table.append([int(action) for action in row.strip()])

environment = RecordVideo(
    make("MountainCar-v0", render_mode="rgb_array"),
    video_folder="./videos",
    episode_trigger=lambda _: True,
)

for _ in range(5):
    state = environment.reset()[0]

    while True:
        position, velocity = state
        discrete_position = discretize_position(position, len(policy_table))
        discrete_velocity = discretize_velocity(velocity, len(policy_table[0]))

        action = int(policy_table[discrete_position][discrete_velocity])
        state, reward, terminated, truncated, _ = environment.step(action)

        if terminated or truncated:
            break

environment.close()
