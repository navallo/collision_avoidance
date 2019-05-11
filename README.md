# Collision Avoidance Env
gym-like collision avoidance environment with ORCA  
 ![image]( https://github.com/navallo/collision_avoidance/blob/master/demo.gif =250x250)
# To reproduce the paper   

## Environment  
Tested on Ubuntu 18.04 and Python 3.6+

## Known dependencies   
 
1. [Optimal Reciprocal Collision Avoidance](https://github.com/sybrenstuvel/Python-RVO2/)  

## Install
```
git clone git@github.com:navallo/collision_avoidance.git
cd collision_avoidance
pip install -e .
```

## Run
run ALAN just online learning
```
cd collision_avoidance/ALAN
python ALAN_true.py
```

run ALAN with action selction
```
cd collision_avoidance/ALAN
python Train_ALAN_action_space.py
```

# To Train using RL (experimental)

## Known dependencies   

1. [OpenAI Gym](https://github.com/openai/gym) (minimum install should be enough)   
 
2. [Optimal Reciprocal Collision Avoidance](https://github.com/sybrenstuvel/Python-RVO2/)  

3. RLlib (Ray):   
https://ray.readthedocs.io/en/latest/index.html  
please install 0.7.0 dev from wheel link  
https://ray.readthedocs.io/en/latest/installation.html#trying-snapshots-from-master


## Install
```
git clone git@github.com:navallo/collision_avoidance.git
cd collision_avoidance
pip install -e .
```

## Run
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

## Reference
http://motion.cs.umn.edu/r/ActionSelection/
