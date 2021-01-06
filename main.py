import ray

from tf_reinforcement_testcases import deep_q_learning, misc


def check_halite_agent(model):
    from kaggle_environments import make

    board_size = 5
    starting_halite = 5000
    env = make('halite',
               configuration={"size": board_size,
                              "startingHalite": starting_halite},
               debug=True)
    trainer = env.train([None])
    obs = trainer.reset()

    halite_agent = misc.get_halite_agent(model)
    return halite_agent(obs, env.configuration)


def one_call(env_name):
    agent = deep_q_learning.RegularDQNAgent(env_name)
    # agent = deep_q_learning.RegularDQNAgent(env_name)
    # agent = deep_q_learning.FixedQValuesDQNAgent(env_name)
    # agent = deep_q_learning.DoubleDQNAgent(env_name)
    # agent = deep_q_learning.DoubleDuelingDQNAgent(env_name)
    # agent = deep_q_learning.PriorityDoubleDuelingDQNAgent(env_name)

    weights, mask, reward = agent.train(iterations_number=1000)
    print(f"Reward is {reward}")


def multi_call(env_name):
    ray.init()
    parallel_calls = 10
    agents = [deep_q_learning.RegularDQNAgent.remote(env_name) for _ in range(parallel_calls)]
    futures = [agent.train.remote(iterations_number=10000) for agent in agents]
    output = ray.get(futures)
    for weights, mask, reward in output:
        print(f"Reward is {reward}")
        misc.plot_2d_array(weights[0], "zero_lvl_with_reward_" + str(reward))
        misc.plot_2d_array(weights[2], "frst_lvl_with_reward_" + str(reward))


if __name__ == '__main__':
    cart_pole = 'CartPole-v1'
    halite = 'gym_halite:halite-v0'
    one_call(cart_pole)
