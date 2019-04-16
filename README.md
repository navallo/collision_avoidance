# Collision Avoidance Env
gym-like collision avoidance environment with ORCA


# Known dependencies   

1. [OpenAI Gym](https://github.com/openai/gym) (minimum install should be enough)   
 
2. [Optimal Reciprocal Collision Avoidance](https://github.com/sybrenstuvel/Python-RVO2/)  

3. RL-baseline:  
https://github.com/hill-a/stable-baselines

# Install
```
git clone git@github.com:navallo/collision_avoidance.git
cd collision_avoidance
pip install -e .
```

# Run
Training and testing:
```
user# cd collision_avoidance/
user# python run_rllib.py
```

just ORCA policy (not training):   
```
user# cd collision_avoidance/collision_avoidance/envs/
user# python collision_avoidence_env.py
```
Tested with python3.7, Ubuntu 18

# Reference
http://motion.cs.umn.edu/r/ActionSelection/
