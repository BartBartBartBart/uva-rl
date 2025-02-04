import numpy as np
from collections import defaultdict

def policy_eval_v(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a all 0 value function
    V = np.zeros(env.nS)
    
    change = np.inf
    
    while change > theta:
        change = 0
        for s in range(env.nS):
            v = 0
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += policy[s][a] * prob * (reward + discount_factor * V[next_state])
            change = max(change, np.abs(v - V[s]))
            V[s] = v

    return np.array(V)

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    # Policy Evaluation
    v = policy_eval_v(policy, env, discount_factor)

    # Policy Improvement
    policy_stable = False

    while not policy_stable:
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * v[next_state])
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_action]
            

        v = policy_eval_v(policy, env, discount_factor)
        

    return policy, V

def policy_iter_v(env, policy_eval_v=policy_eval_v, discount_factor=1.0):
    """
    Policy Iteration Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_v: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    policy_stable = False

    while not policy_stable:

        # Policy Evaluation
        V = policy_eval_v(policy, env, discount_factor)

        # Policy Improvement
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_action]        

    return policy, V

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """

    # Start with a all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    
    change = np.inf

    while change > theta:
        change = 0
        for s in range(env.nS):
            for a in range(env.nA):
                q = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                change = max(change, np.abs(q - Q[s][a]))                

    return policy, Q

def value_iter_q(env, theta=0.0001, discount_factor=1.0):
    """
    Q-value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all state-action pairs.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, Q) of the optimal policy and the optimal Q-value function.
    """

    # Start with a all 0 Q-value function
    Q = np.zeros((env.nS, env.nA))
    
    change = np.inf

    while change > theta:
        change = 0
        for s in range(env.nS):
            for a in range(env.nA):
                q = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    q += prob * (reward + discount_factor * np.max(Q[next_state]))
                change = max(change, np.abs(q - Q[s][a]))  
                Q[s][a] = q              

    return policy, Q
