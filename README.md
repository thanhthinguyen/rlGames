# rlGames

**A video demo for Deep RL agent playing the Tank game**: https://youtu.be/Buawrq0y13k

Deep Reinforcement Learning (Asynchronous Advantage Actor Critic - A3C) agent is trained to play Tank game. Without human expertise, the agent plays in an attacking mode to maximize the score and fails to look after the Base. Therefore the game is over quickly and the score obtained is relatively small. In this demo, the maximum score is 150.

**A video demo for Deep RL agent with a heuristic rule to play the Tank game**: https://youtu.be/syOJn_aCpes

A heuristic rule is incorporated into the training of a deep reinforcement learning (A3C) agent to increase the Tank Battle game score. The agent (green tank) needs to eliminate enemies (red tanks) and protect the Base at the bottom of the game screen. The game is over when either the Base or the agent gets shot. The heuristic rule represents human expertise that provides a playing strategy for the agent depending on the number of red tanks available in the game. When the number of red tanks is more than 2, the agent needs to defend the Base, i.e. defense mode. When the number of red tanks is 2 or less, the agent turns to an attacking mode to eliminate red tanks as quickly as possible. The defense mode implies that the agent needs to focus on the bottom area to protect the Base to prolong the game. In the attacking mode, the agent can go around the entire game screen to look for and eliminate red tanks. The maximum number of red tanks is 5 and the minimum number is 1. Each red tank eliminated will give the agent a score of 10. In this experiment, we demo with only one agent and the score obtained is 550.
