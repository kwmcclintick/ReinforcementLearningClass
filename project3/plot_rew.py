import numpy as np
import matplotlib.pyplot as plt




avg_clipped_reward = np.load('kwm_prj3_rewards_per_episode.npy')


N = 30
episodes = np.linspace(N, len(avg_clipped_reward), len(avg_clipped_reward) - N + 1)
cumsum, moving_aves = [0], []

for i, x in enumerate(avg_clipped_reward, 1):
    cumsum.append(cumsum[i-1] + x)
    if i>=N:
        moving_ave = (cumsum[i] - cumsum[i-N])/N
        #can do stuff with moving_ave here
        moving_aves.append(moving_ave)

print(len(episodes))
print(len(moving_aves))



# plt.plot(episodes, avg_clipped_reward[N-1:])
plt.plot(episodes, moving_aves)
# plt.legend(['Each episode', 'Sliding Avg (N=30)'])
plt.xlabel('Episode')
plt.ylabel('Avg clipped reward (N=30)')
plt.ylim([0, 2])
plt.show()


