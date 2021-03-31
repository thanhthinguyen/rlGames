This project aims to utilise human expertise to extend the capabilities of deep reinforcement learning. 

We modified the Tank game, developed in our works published in [1, 2] to demonstrate the superiority of human expertise incorporation into deep RL training. These codes include the implementation of the Tank game and several deep RL methods, including asynchronous advantage actor-critic (A3C) and deep Q-network (DQN).

![image](https://user-images.githubusercontent.com/59295532/113085936-71c6c380-922c-11eb-8748-dc629607f344.png)

In this Tank Battle game, there is a Base at the bottom of the game screen that deep RL agents need to protect. We can have one or two deep RL agents during the game (green and yellow tanks). The agents need to cooperate in some way to protect the Base and eliminate as many enemies (red tanks) as possible. The game is over if two deep RL agents are eliminated or the Base gets shot by the red tanks. The agent(s) gets a score of 10 when each red tank is eliminated. The game screen is bounded by hard walls (in grey) that the tanks and their bullets cannot move through. There is a soft wall in brown colour where the tanks cannot go through but their bullets can destroy it. The game screen also includes a pond where the tanks cannot move through, but their bullets can go through it. The red tanks are generated randomly and their moving speed increases towards the end of the game to make the game more dynamic. Initially, there are five red tanks, and when there is only one red tank left, four other red tanks will be generated.

The human knowledge shows that when there are many enemies in the game, the Base gets shot quickly and the game is over and thus the score is low. In that situation, the agent needs to defend the Base to prolong the game. When the number of enemies is small, the agent can turn to an attacking mode to eliminate enemies as quickly as possible. This can help to maximize the game score. The heuristic rule therefore is that if the number of red tanks (enemies) is more than 2, the deep RL agent just needs to focus on protecting the Base by working within the bottom area of the game (defense). When the number of red tanks is 2 or less than that, the agent can move anywhere in the entire game area (attacking). Using this heuristic rule, the agent can switch between defense and attacking strategies automatically during the game depending on the status of the game, i.e., the number of enemies.

In the experiments, we compare two scenarios: the first scenario is without human expertise (not using a heuristic rule) and the second scenario is with human expertise (using a heuristic rule via a target mask). The demo videos below show results of the experiments with one agent.

**The first demo video (https://youtu.be/Buawrq0y13k)** shows when the agent plays the Tank game without human expertise. Without human expertise, the agent plays in an attacking mode to maximize the score and fails to protect the Base. Therefore, the game is over quickly, and the score obtained is relatively small. In this demo, the maximum score obtained is 150.

**The second demo video (https://youtu.be/syOJn_aCpes)** shows when the heuristic rule representing human expertise is incorporated into the training of a deep RL A3C agent to increase the Tank game score. The heuristic rule provides a playing strategy for the agent depending on the number of red tanks available in the game. When the number of red tanks is more than 2, the agent tries to move down to the bottom area to protect the Base, i.e., defense mode. When the number of red tanks reduces to 2 or less than that, the agent automatically turns to an attacking mode to eradicate red tanks as quickly as possible. In the attacking mode, we observe that the agent moves to the upper area and thus it is able to eliminate red tanks quicker. This video shows an example where the agent obtains a score of 550 using the above strategic heuristic rule.

[1] Nguyen T. T., et al. (2019, February). Multi-agent deep reinforcement learning with human strategies. In _2019 IEEE International Conference on Industrial Technology (ICIT)_ (pp. 1357-1362). IEEE.

[2]	Nguyen N. D., et al. (2019). Multi-agent behavioral control system using deep reinforcement learning. _Neurocomputing_, 359, 58-68.
