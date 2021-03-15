from rlGames.agents.factory import AgentFactory
from rlGames.configs.a3c import AtariA3CConfig
from rlGames.configs.a3c_map import A3CMapConfig
from rlGames.envs.ale import ALEEnvironment
from rlGames.envs.games.tank_battle.engine import TankBattle
from rlGames.envs.juice import FruitEnvironment, RewardProcessor
from rlGames.learners.a3c import A3CLearner
from rlGames.learners.map import A3CMapLearner
from rlGames.networks.policy import PolicyNetwork
from rlGames.state.processor import AtariProcessor
import numpy as np


class TankBattleTotalRewardProcessor(RewardProcessor):
    def get_reward(self, rewards):
        return np.sum(rewards)

    def clone(self):
        return TankBattleTotalRewardProcessor()

    def get_number_of_objectives(self):
        return 1


def update_reward(rewards):
    return rewards[2]


# switch is combined,


game_engine = TankBattle(render=False, player1_human_control=False, player2_human_control=False,
                         two_players=True, speed=1000, frame_skip=5, debug=False,
                         using_map=True, num_of_enemies=5, min_enemies=2, multi_target=False, switch=True, strategy=2)

env = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor())

network_config = A3CMapConfig(env, initial_learning_rate=0.004)

network = PolicyNetwork(network_config, using_gpu=True, max_num_of_checkpoints=10,
                        load_model_path="./train/tank_battle/a3c_map_checkpoints_03-10-2021-12-20/model-1001370")

# for evaluation epoch 1 and steps are set to 10000,  number of learners 1
agent = AgentFactory.create(A3CMapLearner, network, env, num_of_epochs=1, steps_per_epoch=10000,
                            log_dir='./train/tank_battle/a3c_map_checkpoints',
                            network_update_steps=4, update_reward_fnc=update_reward, num_of_learners=1)

agent.evaluate()
