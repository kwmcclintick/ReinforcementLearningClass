#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import sys

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)




class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # initialize arguements
        self.buffer_size = args.buffer_size
        self.learn_start = args.learn_start
        self.update_target = args.update_target
        self.batch_size = args.batch_size
        self.n_episodes = args.n_episodes

        # initialize non-arguements
        self.env = env
        self.nA = 4  # number of actions
        self.epsilon = 1.
        self.gamma = 0.99

        # replay buffer inits
        # "Once a bounded length deque is full, when new items are added, a corresponding number of items are discarded
        # from the opposite end", so append should suffice for the push function
        self.memory = deque(maxlen=self.buffer_size)

        # initialize nns
        self.DQN = DQN().to(device)
        self.target_DQN = DQN().to(device)
        self.optimizer = optim.Adam(self.DQN.parameters(), lr=args.learning_rate)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            self.DQN.load_state_dict(torch.load('./kwm_prj3_trained_model'))
            # during testing always exploit, never explore
            self.epsilon = 0.

            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        if np.random.rand(1) < (1 - self.epsilon):
            # normalize observtions
            observation = observation / 255.
            # reshape from [84, 84', 4] into a [84, 84', 4, 1] (batch size 1)
            observation = np.expand_dims(observation, axis=-1)
            # make into a torch.cuda.tensor to match DQN weight's data type
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
            # reshape into a [1, 4, 84, 84']
            observation = observation.permute(3, 2, 0, 1)
            # detach so no gradient is required, then make numpy array for numpy argmax() (wont do tensors)
            action = np.argmax(self.DQN(observation).detach().tolist())
        else:
            action = np.random.randint(0, self.nA)
        ###########################
        return action

    def push(self, sasr):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # deque pop()'s an item if full so append is sufficient
        self.memory.append(sasr)
        ###########################


    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        mini_batch = random.sample(self.memory, k=self.batch_size)
        ###########################
        return mini_batch




    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # store learn_start number of (s,a,s',r) values in buffer before training. learn_start must be equal to or
        # greater than mini batch size!
        print('Pre-training, filling replay buffer')
        fill_buffer = True
        i = 0  # time steps taken over all episodes in pretraining
        while fill_buffer:
            ob = self.env.reset()
            done = False
            # get an action from policy
            a = self.make_action(ob)
            # iterate through time steps
            while not done:
                i += 1  # increment time step
                # return a new state, reward and done
                new_ob, r, done, info = self.env.step(a)
                # observe and add (s,a,r,s') to replay buffer
                self.push((ob, a, new_ob, r))

                # make a new action and save new state as old state
                ob = new_ob
                a = self.make_action(ob)

                # check to see if its time to start training and replay buffer is full enough
                if i >= self.learn_start:
                    done = True
                    fill_buffer = False

        print('Replay Buffer size: ', np.shape(self.memory))
        print('Pre-training complete, beginning training')

        ####### TRAINING #######
        # structure from tutorial for mini_batches
        Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

        # track avg clipped reward per episode for plotting
        rew_per_episode = np.zeros(self.n_episodes,)

        # running count of all time iterations
        t_global = 0.
        # begin episode, i = episode #
        for i in range(self.n_episodes):
            # initialize the environment
            ob = self.env.reset()
            done = False
            # get an action from policy
            a = self.make_action(ob)

            # iterate through time steps
            while not done:
                t_global += 1.
                # linearly decaying epsilon from 1. to 0.025 in 1e-5 increments each time step
                if self.epsilon > .025:
                    self.epsilon = self.epsilon - 1e-5

                # return a new state, reward and done
                new_ob, r, done, _ = self.env.step(a)
                # add reward to total
                rew_per_episode[i] += r
                # observe and add (s,a,r,s') to replay buffer
                self.push((ob, a, new_ob, r))
                # sample mini batch uniformly from replay buffer but grouping by states, actions, rewards, newstates
                mini_batch = Transition(*zip(*self.replay_buffer()))

                # for DQN and target DQN to perform forward pass on states, must be unsigned int 8-bit of tensor type
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, mini_batch.next_state)), device=device,
                                              dtype=torch.uint8)

                non_final_next_states = torch.tensor(np.divide([s for s in mini_batch.next_state if s is not None], 255.),
                                                     dtype=torch.float32)
                # reshape states from [32, 84, 84', 4] to [32, 4, 84, 84']
                non_final_next_states = non_final_next_states.permute(0,3,1,2).to(device)

                states = torch.tensor(np.divide(mini_batch.state, 255.), dtype=torch.float32)
                # reshape states from [32, 84, 84', 4] to [32, 4, 84, 84']
                states = states.permute(0,3,1,2).to(device)
                rewards = torch.tensor(mini_batch.reward)
                actions = torch.tensor(mini_batch.action)

                # Compute Q(s_t, a)
                state_action_values = self.DQN(states).gather(1, actions.unsqueeze(1)).squeeze()

                # Compute V(s_{t+1}) for all next states
                next_state_values = torch.zeros(self.batch_size, device=device)
                next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).max(1)[0].detach()
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * self.gamma) + rewards

                # Compute Huber loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

                # reset gradients to zero
                self.optimizer.zero_grad()
                # compute gradients
                loss.backward()
                # clip gradients
                for gradient in self.DQN.parameters():
                    gradient.grad.data.clamp_(-1, 1)
                # apply gradients updating weights
                self.optimizer.step()

                # if its been 5000 time steps, update the target
                if t_global % self.update_target == 0:
                    self.target_DQN.load_state_dict(self.DQN.state_dict())
                    # periodically print training status
                    print('Episode: ', i, ' of: ', self.n_episodes)
                    print('Total time steps: ', t_global)
                    print('Epsilon (chance to explore): ', self.epsilon)
                    print('Average reward/episode last 30 episodes:', np.mean(rew_per_episode[i-30:i]))
                    sys.stdout.flush()

                # update observation
                ob = new_ob
                # choose new action
                a = self.make_action(ob)
                # start new time step
            # start new episode

        # all episodes done
        print('Training complete, saving model, please wait...')
        # SAVE MODEL
        torch.save(self.DQN.state_dict(), './kwm_prj3_trained_model')
        # save plot data
        np.save('kwm_prj3_rewards_per_episode', rew_per_episode)
        print('done')
        ###########################
