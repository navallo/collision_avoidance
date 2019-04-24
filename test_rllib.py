import argparse
import gym
import random
import json

import tensorflow as tf
import tensorflow.contrib.slim as slim

import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.tests.test_multi_agent_env import MultiCartpole
from ray.tune.registry import register_env
from ray.rllib import rollout

import collision_avoidance
from collision_avoidance.envs.collision_avoidence_env import Collision_Avoidance_Env

def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.")

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--steps", default=10000, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")

    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--scenario", type=str, default="crowd")
    
    return parser

class CustomModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        # Example of (optional) weight sharing between two different policies.
        # Here, we share the variables defined in the 'shared' variable scope
        # by entering it explicitly with tf.AUTO_REUSE. This creates the
        # variables for the 'fc1' layer in a global scope called 'shared'
        # outside of the policy's normal variable scope.
        with tf.variable_scope(
                tf.VariableScope(tf.AUTO_REUSE, "shared"),
                reuse=tf.AUTO_REUSE,
                auxiliary_name_scope=False):
            last_layer = slim.fully_connected(
                input_dict["obs"], 64, activation_fn=tf.nn.relu, scope="fc1")
        last_layer = slim.fully_connected(
            last_layer, 32, activation_fn=tf.nn.relu, scope="fc2")
        output = slim.fully_connected(
            last_layer, num_outputs, activation_fn=None, scope="fc_out")
        return output, last_layer

register_env("collision_avoidance", lambda _: Collision_Avoidance_Env(args.num_agents,args.scenario))
ModelCatalog.register_custom_model("model", CustomModel)


parser = create_parser()
args = parser.parse_args()
rollout.run(args, parser)