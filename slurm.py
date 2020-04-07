from gym import envs
from all.experiments import SlurmExperiment
from env import ProcgenAtariEnv
from a2c import a2c

envs = [
    'bigfish',
    'bossfight',
    'caveflyer',
    'chaser',
    'climber',
    'coinrun',
    'dodgeball',
    'fruitbot',
    'heist',
    'jumper',
    'leaper',
    'maze',
    'miner',
    'ninja',
    'plunder',
    'starpilot',
]

device = 'cuda'
envs = [ProcgenAtariEnv(env, device=device) for env in envs]
SlurmExperiment(a2c(device=device), envs, 200e6, sbatch_args={
    'partition': '1080ti-long'
})
