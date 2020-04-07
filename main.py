import argparse
from all.environments import AtariEnvironment
from all.experiments import run_experiment
from all.presets import atari
from env import ProcgenAtariEnv


def run_procgen():
    parser = argparse.ArgumentParser(description="Run an Atari benchmark.")
    parser.add_argument("env", help="Name of the procgen env.")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=10e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", type=bool, default=False, help="Render the environment."
    )
    args = parser.parse_args()

    env = ProcgenAtariEnv(args.env, device='cuda')
    agent_name = args.agent
    agent = getattr(atari, agent_name)

    run_experiment(agent(device=args.device, last_frame=args.frames), env, args.frames, render=args.render)


if __name__ == "__main__":
    run_procgen()
