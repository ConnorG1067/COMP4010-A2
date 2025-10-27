import numpy as np

def QLearning(env, gamma, step_size, epsilon, max_episode):
    # q = np.random.rand(env._n_states, env._n_actions)
    q = np.ones((env._n_states, env._n_actions))

    q[env._goal, :] = 0

    for epsiode in range(max_episode):
        # Initialize s
        s, _ = env.reset()

        converged = False
        while not converged:

            # if epsilon is chosen randomly pick a
            if(np.random.rand() < epsilon):
                a = np.random.randint(env._n_actions)
            # Otherwise pick the best option available
            else:
                a = np.argmax(q[s, :])

            next_s, reward, terminated, truncated, _ = env.step(a)

            q[s,a] = q[s,a] + (step_size * (reward + (gamma * np.max(q[next_s, :])) - q[s,a]))
            s = next_s

            if(s == env._goal):
                converged = True

    # Extract Pi from Q
    Pi = np.zeros((env._n_states, env._n_actions))
    best_actions = np.argmax(q, axis=1)
    Pi[np.arange(env._n_states), best_actions] = 1.0
    
    # Diagonalize
    A = Pi.reshape([env._n_states, env._n_actions])
    return block_diag(*list(A)), q.flatten()

def block_diag(*arrs):
    """
    Create a block diagonal array from provided arrays.

    For example, given 2-D inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : array_like
        Input arrays.  A 1-D array or array_like sequence of length ``n`` is
        treated as a 2-D array with shape ``(1, n)``. Any dimensions before
        the last two are treated as batch dimensions; see :ref:`linalg_batch`.

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal of the last two
        dimensions. `D` has the same dtype as the result type of the
        inputs.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Empty sequences (i.e., array-likes of zero size) will not be ignored.
    Noteworthy, both ``[]`` and ``[[]]`` are treated as matrices with shape
    ``(1,0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> P = np.zeros((2, 0), dtype='int32')
    >>> block_diag(A, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(A, P, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    batch_shapes = [a.shape[:-2] for a in arrs]
    batch_shape = np.broadcast_shapes(*batch_shapes)
    arrs = [np.broadcast_to(a, batch_shape + a.shape[-2:]) for a in arrs]
    out_dtype = np.result_type(*[arr.dtype for arr in arrs])
    block_shapes = np.array([a.shape[-2:] for a in arrs])
    out = np.zeros(batch_shape + tuple(np.sum(block_shapes, axis=0)), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(block_shapes):
        out[..., r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out    


def DynaQ(env, gamma, step_size, epsilon, max_episode, max_model_step):
    q = np.ones((env._n_states, env._n_actions))
    reward_model = np.ones((env._n_states, env._n_actions))
    next_state_model = -1 * np.ones((env._n_states, env._n_actions), dtype=int)

    q[env._goal, :] = 0


    for episode in range(max_episode):
        s, _ = env.reset()
        converged = False
        while not converged:
            if np.random.rand() < epsilon:
                a = np.random.randint(env._n_actions)
            else:
                a = np.argmax(q[s, :])

            next_s, reward, terminated, truncated, _ = env.step(a)

            q[s, a] = q[s, a] + step_size * (reward + gamma * np.max(q[next_s, :]) - q[s, a])


            reward_model[s, a] = reward
            next_state_model[s, a] = int(next_s)

            # Planning Update
            previouslySeenPairs = np.argwhere(next_state_model != -1)
            for _ in range(max_model_step):
                seen_index = np.random.randint(len(previouslySeenPairs))
                state, action = previouslySeenPairs[seen_index]
                r_p = reward_model[state, action]
                s_p = int(next_state_model[state, action])

                q[state, action] = q[state, action] + step_size * (r_p + gamma * np.max(q[s_p, :]) - q[state, action])

            s = next_s
            if terminated or s == env._goal:
                converged = True

    Pi = np.eye(env._n_actions)[np.argmax(q, axis=1)]
    
    return block_diag(*list(Pi)), q.flatten()

def runQLExperiments(env):
    def repeatExperiments(gamma=0.9, step_size=0.1, epsilon=0.1, max_episode=500):
        n_runs = 5
        RMSE = np.zeros([n_runs])
        for r in range(n_runs):
            Pi, q = QLearning(env, gamma, step_size, epsilon, max_episode)
            RMSE[r] = rmse(q, q_star)

        return np.average(RMSE)
    
    # TODO: compute and return the *average* RMSE over runs
    step_size_list = [0.1, 0.2, 0.5, 0.9]
    epsilon_list = [0.05, 0.1, 0.5, 0.9]
    max_episode_list = [50, 100, 500, 1000]
    step_size_results = np.zeros([len(step_size_list)])
    epsilon_results = np.zeros([len(epsilon_list)])
    max_episode_results = np.zeros([len(max_episode_list)])
    
    q_star = np.load('optimal_q.npy')
    # TODO: Set the following random seed to your *student ID*
    np.random.seed(101231686)
    # TODO: Call repeatExperiments() with different step_size in the step_size_list,
    # *while fixing others as default*. Save the results to step_size_results.
    for x in range(len(step_size_list)):
        step_size_results[x] = repeatExperiments(step_size=step_size_list[x])

    # TODO: Call repeatExperiments() with different epsilon in the epsilon_list,
    # *while fixing others as default*. Save the results to epsilon_results.
    for x in range(len(epsilon_list)):
        epsilon_results[x] = repeatExperiments(epsilon=epsilon_list[x])

    # TODO: Call repeatExperiments() with different max_episode in the max_episode_list,
    # *while fixing others as default*. Save the results to max_episode_results.
    for x in range(len(max_episode_results)):
        max_episode_results[x] = repeatExperiments(max_episode=max_episode_list[x])

    return step_size_results, epsilon_results, max_episode_results

def rmse(q, q_star):
    return np.sqrt(np.mean((q-q_star)**2))


def runDynaQExperiments(env):
    def repeatExperiments(gamma=0.9,step_size=0.1,epsilon=0.5,max_episode=100,max_model_step=10):
        n_runs = 5
        RMSE = np.zeros([n_runs])
        for r in range(n_runs):
            Pi, q = DynaQ(env, gamma, step_size, epsilon, max_episode, max_model_step)
            RMSE[r] = rmse(q, q_star)
        return np.average(RMSE)
        
    max_episode_list = [10, 30, 50]
    max_model_step_list = [1, 5, 10, 50]

    results = np.zeros([len(max_episode_list), len(max_model_step_list)])

    q_star = np.load('optimal_q.npy')
    
    # TODO: Set the following random seed to your *student ID*
    np.random.seed(101231686)
    
    for x in range(len(max_episode_list)):
        for y in range(len(max_model_step_list)):
            print("Max Model Step:" , max_model_step_list[y], "Max episode:", max_episode_list[x], "result:", repeatExperiments(max_episode=max_episode_list[x], max_model_step=max_model_step_list[y]))

            results[x][y] = repeatExperiments(max_episode=max_episode_list[x], max_model_step=max_model_step_list[y])

    return results