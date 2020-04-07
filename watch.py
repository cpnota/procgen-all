import argparse
from all.bodies import FrameStack
from all.experiments import GreedyAgent, watch
from env import ProcgenAtariEnv


def watch_atari():
    parser = argparse.ArgumentParser(description="Watch an agent play a procgen environment.")
    parser.add_argument("env", help="Name of the procgen env.")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Playback speed",
    )
    args = parser.parse_args()
    env = ProcgenAtariEnv(args.env, device=args.device)
    agent = FrameStack(GreedyAgent.load(args.dir, env))
    watch(agent, env, fps=args.fps)

if __name__ == "__main__":
    watch_atari()
