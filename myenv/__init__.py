from gym.envs.registration import register

register(
    id ='Firefly-v0',
    entry_point ='myenv.myenv:MyEnv',
)
