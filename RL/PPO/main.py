import gym
import numpy as np
from ppo_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    N = 20 # agent explore 중 N step 쌓이면 learn
    batch_size = 5 # mini batch size to learn
    n_epochs = 4 # learning num at same memory random batch
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, # env로부터 가져옴
                  batch_size=batch_size,
                  alpha=alpha,
                  n_epochs=n_epochs,
                  input_dims=env.observation_space.shape)
    n_games = 100 # game num

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0 # learning num
    avg_score = 0
    n_steps = 0 # agent timestep

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation) # action choose based on cur state(env)
            # print(observation): [-0.00525266 -0.02010498 -0.03208897  0.01877103]
            # print(agent.choose_action(observation)): (1, -0.6783117651939392, 0.05729576200246811)

            observation_, reward, done, info = env.step(action) # get new state from env based on action
            # print(env.step(action)): (array([-0.00994981, -0.40940832, -0.02569036,  0.58367415]), 1.0, False, {})
            n_steps += 1
            score += reward # reward 누적
            agent.remember(observation, action, prob, val, reward, done) # 매 step마다 memory에 trainsition 저장
            if n_steps % N == 0: # game 종료될 때까지 N step마다 learn()
                agent.learn()
                learn_iters += 1
            observation = observation_ # env = new env
        score_history.append(score) # trajectory 종료 시, 누적된 reward인 score 기록
        avg_score = np.mean(score_history[-100:]) # 뒤에서 100번째까지 score 평균값

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)


