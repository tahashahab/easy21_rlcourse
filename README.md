# Background
This project is a blackjack variant card game environment that I wrote in Python to practice different algorithms and concepts that I learned during my independent study of reinforcement learning. All the code is in the repo that shows how I wrote the algorithms for the environment.

## Results

1. Monte-Carlo Control  

The first result from the Monte-Carlo Control algorithm shows the value of each state during the course of the game as the agent traverses the environment. 
The different number of episodes shows that the algorithm learns more about the values of each state, the more trials the agent goes through.
- In 100,000 episodes:

![mcc_100k](https://user-images.githubusercontent.com/46094772/113767200-6243f080-96ec-11eb-8c1d-8899777e6198.png)

- In 500,000 episodes:

![mcc_500k](https://user-images.githubusercontent.com/46094772/113767397-a20ad800-96ec-11eb-8930-15a2b0915bd8.png)

2. Sarsa(λ)

This algorithm uses bootstrapping to update its state-action value function, meaning that is updates after each step unlike Monte-Carlo which updates after each episode.
- Mean-Squared Error for each λ:

![sarsa_mse](https://user-images.githubusercontent.com/46094772/113767941-42f99300-96ed-11eb-9382-63ce9ab83533.png)

- Learning curves for λ=0 and λ=1:

![sarsa_learning_curve](https://user-images.githubusercontent.com/46094772/120738870-922a2d00-c4be-11eb-9077-856e0a37d40a.png)

3. Sarsa(λ) with Linear Function Approximation

Adding function approximation to Sarsa(λ) streamlines the learning process as we are combining states together so it takes less time to go through every option multiple times. 
- Mean-Squared Error for each λ:

![sarsa_mse_lfa](https://user-images.githubusercontent.com/46094772/113767960-47be4700-96ed-11eb-8bb5-05aa2a5175f9.png)

- Learning curves for λ=0 and λ=1:

![sarsa_learning_curve_lfa](https://user-images.githubusercontent.com/46094772/113767975-4a20a100-96ed-11eb-99cd-937ff9edb1f9.png)

## Conclusion

- Bootstrapping with Sarsa(λ) (with and without linear function approximation) is generally much faster than Monte-Carlo Control, however, the way that the environment is set up where rewards are only handed at the end of the epsiode, step-by-step updates (R=0) could lower the value of a good state.
- We are assuming the value function in this case is linear by applying a linear function approximator, but the actual function may be complex.
- It seems as if bootstrapping and not both have their pros and cons, but since MCC has the least complex construction and the results are similar, my conclusion is that it is the better algorithm to explain the value function for this environment. 

