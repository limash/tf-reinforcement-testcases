import ray
# from kaggle_environments import make

from tf_reinforcement_testcases import deep_q_learning, misc

if __name__ == '__main__':
    cart_pole = 'CartPole-v1'
    halite = 'gym_halite:halite-v0'

    ray.init()
    parallel_calls = 10
    agents = [deep_q_learning.RegularDQNAgent.remote(cart_pole) for _ in range(parallel_calls)]
    futures = [agent.train.remote(iterations_number=10000) for agent in agents]
    output = ray.get(futures)
    for weights, mask, reward in output:
        print(f"Reward is {reward}")
        misc.plot_2d_array(weights[0], "zero_lvl_with_reward_" + str(reward))
        misc.plot_2d_array(weights[2], "frst_lvl_with_reward_" + str(reward))

    # agent = deep_q_learning.RegularDQNAgent(cart_pole)
    # agent = deep_q_learning.RegularDQNAgent(cart_pole)
    # agent = deep_q_learning.FixedQValuesDQNAgent(cart_pole)
    # agent = deep_q_learning.DoubleDQNAgent(cart_pole)
    # agent = deep_q_learning.DoubleDuelingDQNAgent(halite)
    # agent = deep_q_learning.PriorityDoubleDuelingDQNAgent(halite)

    # model, reward = agent.train(iterations_number=10000)
    # print(f"Reward is {reward}")

    # board_size = 5
    # starting_halite = 5000
    # env = make('halite',
    #            configuration={"size": board_size,
    #                           "startingHalite": starting_halite},
    #            debug=True)
    # trainer = env.train([None])
    # obs = trainer.reset()

    # halite_agent = misc.get_halite_agent(model)
    # actions = halite_agent(obs, env.configuration)
