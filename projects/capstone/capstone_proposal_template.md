# Machine Learning Engineer Nanodegree
## Capstone Proposal
Chris Wilkins
April, 2017

## Proposal

### Domain Background

The paper by R.S. Sutton and A.G Barto (1) defines Reinforcement Learning as 'learning what to do so as to maximize a numerical reward signal'. 
Reinforcement Learning has been used in order to train computers to play many different games, with increasing success. From TD-Gammon (2) playing Backgammon at a world-class level, 
to the work of Mnih et al (3) playing Atari games, to Google's DeepMind beating the world's leading Go player, reinforcement learning has drastically improved in solving ever-more complex games.

The game chosen for this project is called King's Table, or Hnefatafl (4), and is an ancient Viking board game played between two players. It has similarities to Chess, in that there is a King
and the capture of the King piece signals the end of the game. But it differs since only one side has the King piece, the side with the King (the Defenders) has the goal of getting the King
to one of the four corner squares. The opposing side (the Attackers) has the goal of stopping the King from getting to the corner squares, and can win the game by surrounding the King on all sides
with Attacking pieces. The Defenders have fewer pieces, but an easier goal at first glance, while the Attackers have more pieces but the more difficult task.
This disparity between Attackers and Defenders makes the game interesting from a Machine Learning perspective. It is fairly obvious how to implement a simple algorithm which finds the shortest path to 
one of the corner squares as a strategy for the Defenders, this doesn't require any Machine Learning to achieve success. The strategy for the Attackers is more subtle, being able to block all the exit 
paths is something that inexperienced players struggle with. Can a computer learn this?


### Problem Statement

The goal of this project is to use Reinforcement Learning to make an agent learn to play King's Table as the Attackers to a level where it can beat a beginner level heuristic-based computer player. 
Since the initial agent will probably lose the vast majority of games against a beginner level computer player, the agent will also be measured by the number of rounds it can keep playing before losing.


In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. 
Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , 
measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

Since the agent will be learning from playing against a computer opponent, there are no external datasets. However, given the state space is large, the board is 11x11 and the 
37 pieces (24 Attackers and 13 Defenders) have over 100 choices for valid moves to play at any given game state, it probably won't be possible to fill the entire Q-Table for all states and actions via self-play. 
For this reason, it may be necessary to use Deep Reinforcement Learning to approximate the value function for each state.

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. 
Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary 
It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

### Solution Statement
_(approx. 1 paragraph)_

The agent will use Deep Reinforcement Learning to automatically learn the optimal strategy by playing games against a computer opponent. The agent will try to learn the strategy from scratch,
with no hand-crafted features added to the model.

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. 
Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , 
measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

The computer player will use a shortest-path algorithm to try to find the shortest path to the exit squares. If there are no paths to the exit due to the King being blocked by it's own pieces,
the computer player will try to move pieces out of the way of the King until the King has a free path to the exit. If the Attacking pieces have managed to block all the exits, the computer player
will look for opportunities to capture Attacking pieces. If no such capture opportunities exist, the computer player will select a random move from the valid available moves.

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. 
Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. 
Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

Winning Percentage - The number of games won against the computer opponent will be the first, and most important metric. It will be calculated by running a set number of games against the 
computer and measuring the average number of games won by the agent.

Game Length - As mentioned above, the agent is expected to lose all the games against the computer while learning. In the event that the win percentage is very small, the average length of the 
games played will be used as a secondary metric for how well the agent is learning.


In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. 
The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. 
Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
_(approx. 1 page)_

The project will be implemented in Python3, using the Tensorflow library for the Deep Network. The first step to be implemented will be to design a simulator which will allow the agent
to play games against the computer opponent. The simulator will show the board and the two sets of pieces, and implement the rules of the game with regards to moving pieces and capturing. 
The logic for the computer opponent will be build into the simulator, and will use a shortest-path algorithm to find the best move, as described above.

The simulator should support two modes: a training mode where the agent plays a large number of games against the computer opponent in order to learn, and a testing mode where the agent
plays a set number of games against the computer using the trained model. The training mode should be focussed on exploring the state space, whereas the testing mode should focus on 
choosing the best moves based on the learning so far.


In this final section, summarize a theoretical workflow for approaching a solution given the problem. 
Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. 
The workflow and discussion that you provide should align with the qualities of the previous sections. 
Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. 
The discussion should clearly outline your intended workflow of the capstone project.

### References

1. Reinforcement Learning: An Introduction - http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf
2. Temporal Difference Learning and TD-Gammon - http://www.bkgm.com/articles/tesauro/tdl.html
3. Human Level control through deep Reinforcement Learning - http://files.davidqiu.com/research/nature14236.pdf
4. Hnefatafl: The Game of the Vikings - http://tafl.cyningstan.com/

-----------

**Before submitting your proposal, ask yourself. . .**

- Does the proposal you have written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
