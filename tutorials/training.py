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


def train_tank_1_player_machine():
    game_engine = TankBattle(render=False, player1_human_control=False, player2_human_control=False,
                             two_players=False, speed=2000, frame_skip=5)

    env = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor(),
                           reward_processor=TankBattleTotalRewardProcessor())

    network_config = AtariA3CConfig(env, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config, max_num_of_checkpoints=20)

    agent = AgentFactory.create(A3CLearner, network, env, num_of_epochs=10, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/tank_battle/a3c_checkpoints')

    agent.train()


def train_tank_1_player_machine_with_map():
    def update_reward(rewards):
        return rewards[2]

    game_engine = TankBattle(render=False, player1_human_control=False, player2_human_control=False,
                             two_players=False, speed=1000, frame_skip=5, debug=False,
                             using_map=True, num_of_enemies=5, min_enemines=2, multi_target=True, strategy=0
                             )

    env = FruitEnvironment(game_engine, max_episode_steps=10000, state_processor=AtariProcessor())  # 10000

    network_config = A3CMapConfig(env, initial_learning_rate=0.004)

    network = PolicyNetwork(network_config, using_gpu=True, max_num_of_checkpoints=20)

    agent = AgentFactory.create(A3CMapLearner, network, env, num_of_epochs=10, steps_per_epoch=1e6,
                                checkpoint_frequency=5e5, log_dir='./train/tank_battle/a3c_map_checkpoints',
                                network_update_steps=4, update_reward_fnc=update_reward)

    agent.train()


if __name__ == '__main__':
    train_tank_1_player_machine_with_map()
