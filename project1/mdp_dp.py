### MDP Value Iteration and Policy Iteration
### Reference: https://web.stanford.edu/class/cs234/assignment1/index.html 
import numpy as np

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Evaluate the value function from a given policy.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS,nA]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    
    value_function = np.zeros(nS)
    ############################
    # page 75, iterative policy evaluation of suttons book
    theta = tol

    # LINE 1
    while True:
        # LINE 2
        delta = 0.
        # LINE 3
        for state in range(nS):

            # LINE 4
            V_current = 0  # the new value function for this state

            # LINE 5 sum
            for action in range(nA):  # iterate through actions that can be taken in that state
                for next_state in P[state][action]:
                    P_next = next_state[0]
                    next = next_state[1]
                    reward = next_state[2]
                    terminal = next_state[3]

                    V_current += policy[state, action] * P_next * (reward + gamma * value_function[next]) # calculating sum of V(s) assignment

            # LINE 6
            delta = max(delta, abs(V_current - value_function[state]))  # calculate change in value function for this state
            value_function[state] = V_current

        # LINE 7
        if delta < theta:  # if converged, break loop and move to next value function element
            break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters:
    -----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    Returns:
    --------
    new_policy: np.ndarray[nS,nA]
        A 2D array of floats. Each float is the probability of the action
        to take in that state according to the environment dynamics and the 
        given value function.
    """

    new_policy = np.ones([nS, nA]) / nA
	############################
	# page 80 from textbook
    new_policy = np.zeros([nS, nA])

    for state in range(nS):
        Q = 0
        for action in range(nA):  # iterate through actions that can be taken in that state
            for next_state in range(len(P[state][action])):
                # save tuple values to variables
                P_next = P[state][action][next_state][0]
                next = P[state][action][next_state][1]
                reward = P[state][action][next_state][2]
                terminal = P[state][action][next_state][3]

                new_policy[state, action] += P_next * (reward + gamma * value_from_policy[next])  # calculating sum 2 of V(s) assignment
        # pick highest Q as action of probability 1, all other actions probability 0
        highest_Q_action = np.argmax(new_policy[state, :])
        new_policy[state, :] = 0.
        new_policy[state, highest_Q_action] = 1.

	############################
    return new_policy


def policy_iteration(P, nS, nA, policy, gamma=0.9, tol=1e-8):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: policy to be updated
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    new_policy: np.ndarray[nS,nA]
    V: np.ndarray[nS]
    """
    new_policy = policy.copy()
    ############################
    # (1) Initialization
    # contained in arguements

    while True:
        # (2) Policy Evaluation
        V = policy_evaluation(P, nS, nA, policy, gamma, tol)
        # (3) Policy Improvement
        new_policy = policy_improvement(P, nS, nA, V, gamma)

        if np.array_equal(new_policy, policy):  # check for policy stability
            break
        else:
            policy = new_policy                 # otherwise update old policy and try again
    ############################
    return new_policy, V

def value_iteration(P, nS, nA, V, gamma=0.9, tol=1e-8):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    V: value to be updated
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    policy_new: np.ndarray[nS,nA]
    V_new: np.ndarray[nS]
    """

    V_new = V.copy()
    ############################
    policy_new = np.zeros([nS, nA])

    # page 83, iterative policy evaluation of suttons book
    theta = tol


    # LINE 1
    while True:
        # LINE 2
        delta = 0.
        # LINE 3
        for state in range(nS):

            # LINE 4
            V_max = 0  # the new value function for this state
            highest_Q_action = 0

            # LINE 5 sum
            for action in range(nA):  # iterate through actions that can be taken in that state
                V_current = 0
                for next_state in P[state][action]:
                    P_next = next_state[0]
                    next = next_state[1]
                    reward = next_state[2]
                    terminal = next_state[3]

                    V_current += P_next * (reward + gamma * V_new[next])  # calculating sum of V(s) assignment
                if V_current > V_max:
                    V_max = V_current
                    highest_Q_action = action

            # LINE 6
            delta = max(delta, abs(V_max - V_new[state]))  # calculate change in value function for this state
            V_new[state] = V_max

            policy_new[state, :] = 0.
            policy_new[state, highest_Q_action] = 1.

        # LINE 7
        if delta < theta:  # if converged, break loop and move to next value function element
            break



    ############################
    return policy_new, V_new

def render_single(env, policy, render = False, n_episodes=100):
    """
    Given a game envrionemnt of gym package, play multiple episodes of the game.
    An episode is over when the returned value for "done" = True.
    At each step, pick an action and collect the reward and new state from the game.

    Parameters:
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as attributes.
    policy: np.array of shape [env.nS, env.nA]
      The action to take at a given state
    render: whether or not to render the game(it's slower to render the game)
    n_episodes: the number of episodes to play in the game. 
    Returns:
    ------
    total_rewards: the total number of rewards achieved in the game.
    """
    total_rewards = 0
    for _ in range(n_episodes):
        ob = env.reset() # initialize the episode
        done = False
        while not done:

            if render:
                env.render() # render the game
            ############################
            # agent performs an action
            a = np.argmax(policy[ob,:])
            # game returns a reward and new state
            ob, r, done, info = env.step(a)
            # add reward to total
            total_rewards += r
            ############################
            
    return total_rewards



