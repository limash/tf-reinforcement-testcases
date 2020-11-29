from kaggle_environments import make

from tf_reinforcement_testcases import deep_q_learning, misc

if __name__ == '__main__':
    cart_pole = 'CartPole-v1'
    halite = 'gym_halite:halite-v0'

    # agent = deep_q_learning.RegularDQNAgent(cart_pole)
    # agent = deep_q_learning.FixedQValuesDQNAgent(cart_pole)
    # agent = deep_q_learning.DoubleDQNAgent(cart_pole)
    agent = deep_q_learning.DoubleDuelingDQNAgent(halite)
    model = agent.train(iterations_number=10000)

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
