# Reinforcement-Learning

## Q-Learning

### Introduction
In this we build a Reinforcement Learning Environment based on Markov
decision process. The environment is a Grid World. Both deterministic as
well as non-deterministic (stochastic) environments have been developed
using a set of states, actions and rewards. The stochastic environment has
been built using a transition probability matrix. The environment is solved
using the Q-learning tabular method.

The Environment that is built over here is a Grid World Environment. This environment has in
total 25 states (5 * 5 matrix). We have two terminal states: Goal state and Lost State. The Goal
state is positioned at (4,4) (Green Color Box) and the Lost state is positioned at (4,2) (Blue
Color Box). Our agent is at position (0,0) (denoted by Yellow box) which
is the starting state. The agent can take any of the four actions: Up, Down, Left and Right in
any state. When the agent reaches the Goal state it receives an immediate reward of +4, when
it reaches the Lost state it receives an immediate reward of -3 and receives an immediate
reward of -1 when it goes to any other state. The main objective of this project is to make the
agent (at (0,0)) learn the Grid World environment using the Q-learning tabular method and
make it reach the Goal State in both Deterministic as well as Stochastic environment.

### Stochastic vs Deterministic Environment
In this project we develop both Deterministic as well as Non-Deterministic (Stochastic)
environments. The main difference between them is, in a deterministic environment if an
agent is currently at some position and selects an action, the selected action completely
determines the next state of the environment. But in the case of a stochastic environment, it is
random in nature and cannot be completely determined by the agent. For example, let’s say an
agent is in state A and the action taken is Left. In the case of a deterministic environment, the
agent will definitely go to the next state B (let’s say) without any uncertainty. But in case of
stochastic environment since there is some randomness, it may not go to state B but may go to
another state C. In this project we use agent action to be non-deterministic to represent the
stochastic nature of the environment.

### Conclusion
We were able to see how using Q-learning we were able to solve the deterministic as
well as the stochastic environment. After trying out various parameters for discount factor
gamma, learning rate alpha, epsilon value we found using 0.8 as discount factor was better and
also the values of 0.2 and 0.99 as the learning rate and the epsilon in updating the values in the
q table helped the agent to gain more rewards thus able to converge to the winning state. The
stochastic nature of the stochastic environment can also be interpreted from the graphs as you
can observe some differences in the number of timesteps and the rewards when compared with
that of deterministic environment, and finally the stochastic environment behaves similar to
the deterministic environment.



## Value Function Approximation

### Introduction
In this we explore two Open AI Gym Environments and implement value
function approximation algorithms on these environments. We use Open AI
CartPole and Open AI Atari Breakout environments. The Deep Q Learning
algorithm was used where the neural networks acted as a function
approximator. This algorithm was implemented following Deep Mind’s
paper that played Atari from raw pixels. We first implement Vanilla DQN
and apply it on the above-mentioned environments. Then, we implement
Dueling DQN, an improvement to the DQN algorithm, on the two
environments.

The first environment that was used is the Open AI CartPole. It consists of a pole that is
attached by an un-actuated joint to a cart, that moves along a frictionless track. Here the cart
acts as the agent. The system is controlled by applying a force or push of -1 or +1 to the cart.
The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is
provided for every timestep that the pole remains upright. The episode ends when the pole
deviates more than 15 degrees from vertical, or the cart is more than 2.4 units from the center
(i.e., the cart goes out of screen). It has four states: cart position (ranging from -2.4 to +2.4
units), cart velocity (-∞ to +∞), pole angle (~-41.8 o to +41.8 o ) and pole velocity at tip (-∞ to
+∞). It has two actions: push the cart to left or right. The maximum episode length is 200 and
it terminates after that. The neural networks are used as function approximators.

The second environment that was used is the Open AI Atari Breakout. In this environment the
observation or state is an RGB image of the screen which is an array of shape (210,160,3). It
comprises four actions left, right, fire and No-op. The agent here is the bottom paddle that
controls the game. We use a Deterministic v4 environment (related to skipping 4 frames). We
stack four recent consecutive frames or observations as input to the algorithm to know the
direction, velocity and acceleration of the ball. Since the environment has too many pixels, we
reduce or crop the size of the environment and also, we convert the RGB fame to grayscale
and store the pixels in uint8 format to reduce the size. The reward here +1 is for each time the
ball hits the block and the game is finished once the ball does not hit the paddle for a total of
five times (the game has five lives). We pre-process the states of this game using CNN and
use neural networks as a function approximator.


### Deep Q Learning

In this project we use Deep Q-learning to solve the environment. The main drawback of using
Q learning is if the combinations of states and actions are too large, the memory and the
computation requirement for Q will be too high. To address this, we use a function
approximator that generalizes the approximation of the Q-value function rather than
remembering the whole solution. This is the main advantage of representing the Q function as
q(s; w) where it represents the function approximator (state s and weight parameter w). The
DQNs use a deep neural network to estimate Q-values directly from state information . The
model outputs one Q-value for each action, and we select the action which gives the highest
Q-value in order to perform the best in the environment. The model learns by calculating the
loss between our future reward and the reward the model expected when in a given state. We
make use of an epsilon-greedy method for choosing actions (Exploration and exploitation).
Exploration is to explore the environment to find better solutions and exploitation uses the
explored paths to exploit the best solution. When the agent exploits the state, it will take the
action that maximizes the Q value according to current estimation of Q value. We also make
use of epsilon decay to balance exploration and exploitation.

But the problem in Reinforcement Learning is that if we feed successive samples of inputs to
the neural network model they are correlated and data may not be i.i.d. (independent and
identically distributed). In this case the model may be overfitted and the solutions may not be
generalized. In addition, in supervised learning the output label does not change over time but
in reinforcement learning the targets(label) change thus making training unstable.
Thus, we come up with solutions to address this. The first one is called Experience Replay. In
this we have a buffer wherein we store a lot of transactions (here one transaction comprises
the current state, action, next state, reward and done). Every time we sample a mini-batch of
sample size n (here we use 32) from this buffer to train the network. This forms the input
dataset for our training model. Since we sample data from replay memory, the data is
independent and follows i.i.d. Without the use of replay memory, the model does not learn
properly and may go into local minima. The second solution to changing targets or labels is
the use of a target network. We create two different networks w and w - . The first network is
used to retrieve Q values while the second network keeps track of updates in the training. We
synchronize the weights of both networks after some number of steps or episodes. The
purpose of this is to fix the Q-value targets temporarily so we don’t have a moving target to
chase.

### Dueling Deep Q Learning

It is an improvement over the Vanilla Deep Q-Learning. We know that Q-value represents the
value of choosing a specific action in a given state, and
the V- Value represents the value of a state irrespective of the given action. In Dueling DQN,
we decompose the Q-value (i.e., the estimator) into the Value of a state V(s) and Advantage of
doing action A while in state S.
Q(S, a) = V(S)+A(S, a)

The goal of decomposing is that the model can learn which states are less or more valuable
without having to learn the effect of each action in each state. This helps to calculate only the
useful states which are necessary i.e., if a state tends to be bad there is no use in calculating
the value of actions for that state. In the above figure the Dueling DQN architecture has its
output decomposed into Value and Advantage. The Value function is a single value denoting
the value of a particular state while the advantage function has multiple output neurons
denoting the advantage value of each action (in our case four values because of four actions).
Finally, when we aggregate these two layers, we subtract the average advantage from all
possible actions of the state and add it with the value function to give the final Q-value output.
It increases the stability of the optimization and thus converges to optimal values quickly.
Q(S, a) = V(S)+[A(S, a)-(1/|A|)(Σ a’ A(S, a’))]

### Conclusion

We were able to see how using Deep Q-learning was able to perform better than the
normal Q-learning. We should use Deep Q Learning for Complex environments as it does not
need to store any Q-values. This is the main advantage of DQN over Q-Learning. The results
were pretty good. When it comes to Duel DQN the results were much better than vanilla
DQN. You can notice it especially in the case of the Atari Breakout game where the rewards
were higher than when implemented using DQN. So, to conclude, using Duel DQN does
increase the performance and rewards but even the performance of DQN is good but is not
better. Thus, from the plots and the results obtained we can conclude that Duel DQN is better
than Vanilla DQN.

## Policy Gradient Methods

### Introduction
In this we explore the Open AI Gym Environment Cartpole and implement
the policy gradient and actor critic algorithm. Specifically, we are
implementing the policy gradient REINFORCE algorithm and the TD Actor
Critic Algorithm. The goal of the reinforcement learning is to find optimal
behavior strategy for the agent to obtain optimal rewards. The policy gradient
method targets modelling and optimizing the policy directly.

The environment that was used is the Open AI Gym CartPole-v0. It consists of a pole that is
attached by an un-actuated joint to a cart, that moves along a frictionless track. Here the cart
acts as the agent. The system is controlled by applying a force or a push of -1 or +1 to the cart.
The pendulum starts in an upright position, and the goal is to prevent it from falling over. A
reward of +1 is provided for every timestep that the pole remains in upright position.
The episode ends when the pole deviates more than 15 degrees from vertical, or the cart is
more than 2.4 units from the center (i.e., the cart goes out of screen). It has four states: cart
position (ranging from -2.4 to +2.4 units), cart velocity (-∞ to +∞), pole angle (~-41.8 o to
+41.8 o ) and pole velocity at tip (-∞ to +∞). It has two actions: push the cart to left or right. The
maximum episode length is 200 and it terminates after that. We use the policy gradient
methods to reach the goal.

### Policy Gradient

In this project we make use of policy gradient algorithms to solve the environment. Our main
goal is to find an optimal behavior strategy to obtain optimal rewards. The policy gradient
algorithms instead of optimizing the value or the action values, tends to optimize the policy
directly thereby maximizing the rewards. The main advantages of using policy gradient
methods are better convergence, effective in high-dimensional as well as continuous spaces
and can learn stochastic policies. It is usually modeled using a parameterized function respect
to θ , that is π θ (a|s) (which is the policy) , where a is the action and s is the state. The reward
function (objective function) depends on the policy and we can use various algorithms to
optimize θ , for the best reward.

### REINFORCE

The first policy gradient algorithm that we are using is the REINFORCE. It is a Monte Carlo
based policy gradient approach. The policy parameter θ is updated based on the estimated
return by the Monte Carlo method and is updated using the stochastic gradient ascent.

The update is proportional to the return Gt , the gradient of the probability of taking the action
actually taken and is divided by the probability of taking that action. The algorithm works as
follows:
1. First we initialize the policy parameter θ at random.
2. Generate a trajectory on policy π θ : S 1 , A 1 , R 1 , S 2 , A 2 , …, S Tot .
3. For t=1, 2, …, T:
  A. Estimate the the return G t ;
  B. Update policy parameters: θ←θ+αγ t G t ∇ θ log π θ (At|St)

### Actor Critic

In the previous algorithm REINFORCE we saw that it is based on Monte Carlo updates. Due
to this there is high variability in log probabilities and the cumulative rewards will make noisy
gradients and cause unstable learning thus can lead to unstable policy. Intuitively, making the
cumulative reward smaller by subtracting it with a baseline will make gradients smaller, and
thus smaller and more stable updates.

We use the Actor Critic method and in particular I am using TD based Actor
Critic method. In general, in a policy gradient there are two components namely policy model
and the value function. We can make use of value function in addition to policy as it can assist
the policy update and this is what Actor Critic method does.
In TD based actor critic method, it consists of two neural network models. One of them is the
critic model which updates the value function parameters w, i.e., for V(s) and the actor
updates the policy parameters θ for π θ (a|s), in the direction suggested by the critic. In the TD
actor critic method, we first take an action from the present state with the help of the actor and
get the rewards and the next state. The advantage of using the Actor Critic method is it has lower 
variance with no bias.

### Conclusion

We were able to see how using TD Actor Critic was able to perform better than the
REINFORCE algorithm. This is because of the Monte Carlo update of the REINFORCE
algorithm because of which there is a lot of variance in the rewards. The TD based Actor
Critic uses both the Actor network and Critic Network for updating policy and evaluating the
states/actions. The difference in using both the algorithms is not huge in terms of performance
in the cartpole environment since it is not a complex environment. But overall, Actor Critic
methods should have better performance than the REINFORCE algorithm.
