from gym.envs.registration import register

register(
    id='collision_avoidance-v0',
    entry_point='collision_avoidance.envs:Collision_Avoidance_Env',
)
