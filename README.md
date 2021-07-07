# pacman-assignment4
DD2438 Pac-Man

This project is about *UC Berkley’ Pac-man Capture the Flag*.

We developed 2 solutions to control agents:
- deep Q-learning algorithm
- α−β tree search

The __deep Q-learning__ algorithm is inspired by Trapit Bansal, Jakub Pachocki, Szymon Sidor, Ilya Sutskever, and Igor
Mordatch. Emergent complexity via multi-agent competition, 2018.
The two agents play again each other and learn.
We faced difficulties to make the agents learn because we could not run enough games.
We then used a randomized baseline policy as behaviour policy for the agents, which consists of an offensive agent and a defensive agent.
To introduce some exploration, we do play this policy with a chance of 80% and with a chance of 20% we choose a random action.


To come up with a solution at the end of the project we also developed __alpha-beta__.
We used the following features in the heuristic fonction:
* The difference between our score and the opponent score
* The collected food by our team
* The collected food by the opponents
* The sum of the distance from each of our agents to the closest food
* The distance between our agents
* The distance from our main agent to our side
* The distance from the teammate agent to our side
*The distance from our main agent to the closest opponent
