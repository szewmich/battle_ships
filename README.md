# battle_ships
This project is a numerical simulator of a simple "battleships" game which main scope is about  finding, employing and validating a perfect algorithm for playing the game. The goal is winning the games in lowest amount of moves possible, on average. 
To achieve this, the most straightforward assumption is that the player should in each move maximize the probability of scoring a hit - this is what my algorithm does, it calculates probabilities and targets the most promising fields. I think it's fair to say this is one good way of approaching the problem, though not sure if it's "perfect" for achieving the actual end goal (minimize amount of moves per game). More on that may come in future if I decide to explore other paths.
 
The game rules I employed are as follows:
•	Board is 10x10 size
•	Ships to be placed:
o	1 x 5-segment ship
o	1 x 4-segment ship
o	2 x 3-segment ship
o	3 x 2-segment ship
•	Ships CANNOT touch each other (at least one empty space between ships, all sides and diagonally)
•	Ships CAN be placed at board's edges.

These might differ from what you are used to, specially with regards to amount of ships. I have all these rules hardcoded, as I didn't see a point in complicating it to make it flexible. It's just a convention, this project's main idea of  making a perfect algorithm is not really affected. 

The 3 main building blocks of the project are:
1.	Random board setup generator
2.	Various algorithms for determining the optimal move in a given situation 
3.	Multiple games simulator - employs the above 2 and collects statistical data

Other than that and referenced function modules, there are some auxiliary scripts like validation checks, auto-runs, database format converters etc.

The main scripts are widely documented (docstrings and comments).

To explore the project, follow this order:
1. prob_map_montecarlo.py + dependencies
Main function of this script takes a certain board state as an input and calculates the probability matrix for scoring a hit in each field using Monte-Carlo approach and statistical methods for estimation. It relies on other modules: random board generator and general functions module.
2. prob_map_advanced.py + dependencies
Main function of this script takes a certain board state as an input and calculates the probability matrix for scoring a hit in each field using brute force approach (generation and counting of all possible setups). Same dependencies as above. Despite its simplistic approach, this script's build might actually be harder to understand than monte-carlo one, because of a giant, 7-level nested loop that runs through all ships to be placed.
3. game.py + dependencies
Multiple games simulator - generates random games, calculates probabilities with the above methods, saves or retrieves calculation data into the database (so called 'RTP'), shoots the most promising fields, updates the board state and repeats the shooting sequence until all ships get sunk. Finally, it gathers statistics on average hit success rates and required amount of moves to complete the game.
4. param_sens_study.py + dependencies + param_sens_study_notebooks
Study performed in order to determine the optimal parameters for monte-carlo method. It was a tool used to run multiple game scenarios repeatedly and compare results of advanced method vs monte-carlo method with different statistical parameters, in terms of calculation time and accuracy. This is 'data analytics' part of this project - analysis of the results obtained in a few loops of the study are presented in jupyter notebook format.


Current state of the project:
- Param sens study is done, optimal parameters for monte-carlo method are employed.
- Calculation database (RTP) contains above 15k entries.
- Multiple game simulator - the curve of calculation time vs number of entries in RTP is already almost flat, with average game simulation time of 60sec. Average amount of moves to win the game converges to ~44.39

What's coming next:
- module responsible for non-random bias detection in board setups done by "defending" player
- adjustment to shooting algorithm based on above dias detection, leading to an improvement in average moves to win when playing against an opponent who does not deploy ships in ideally rabdom fashion