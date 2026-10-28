import numpy as np
import torch
import torch.nn as nn
import torch.optim
import random
from replay_buffer import replay_buffer
from actor_critic_networks import ActorNetwork, CriticNetwork 


class TD3():
    """Implements TD3"""
    def __init__(self, action_magnitude=2,sigma_exp_noise=.2, N=64, gamma=.99, sigma_tilda=.4, d=2, tau=.005, c=1,lr_critic=.001,lr_actor=.001):

        # Initalize main actor and critics
        self.actor_main=ActorNetwork(action_magnitude)
        self.critic_1_main=CriticNetwork()
        self.critic_2_main=CriticNetwork()

        # Initialzie target networks w/ main network parameters
        self.actor_target=ActorNetwork(action_magnitude)
        self.critic_1_target=CriticNetwork()
        self.critic_2_target=CriticNetwork()

        self.actor_target.load_state_dict(self.actor_main.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1_main.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2_main.state_dict())

        # Create replay buffer for transition storing
        self.buffer=replay_buffer()


        # TD3 paramaters
        self.action_magnitude=action_magnitude
        self.sigma_exp_noise=sigma_exp_noise
        self.N=N
        self.gamma=gamma
        self.sigma_tilda=sigma_tilda
        self.d=d
        self.tau=tau
        self.c=c
        self.lr_critic=lr_critic
        self.lr_actor=lr_actor

        # Critic loss and optimizers for main networks
        self.main_critic_loss_fn=nn.MSELoss()
        self.critic_1_optimizer=torch.optim.Adam(self.critic_1_main.parameters(),lr=lr_critic)
        self.critic_2_optimizer=torch.optim.Adam(self.critic_2_main.parameters(),lr=lr_critic)
        self.main_actor_optimizer=torch.optim.Adam(self.actor_main.parameters(),lr=lr_actor)


        self.step_counter=0

        # Track training losses 
        self.critic_1_loss=[]
        self.critic_2_loss=[]
        self.actor_loss=[]

    def select_action(self,state,explore=True):
        """Select action with or without exploration noise."""
        with torch.no_grad():
            action=self.actor_main(torch.tensor(state,dtype=torch.float32))

            if explore:
            # Add noise for exploration
                random_error=torch.randn_like(action)*self.sigma_exp_noise
                action+=random_error


            # Keep action within action bounds
            action=torch.clamp(action,-self.action_magnitude,self.action_magnitude)
            return action.numpy()

    def store_in_buffer(self,state, action_array, reward,new_state,terminated, truncated):
        """Store env transition in buffer"""
        self.buffer.add(state,action_array,reward,new_state,terminated,truncated)


    def train(self):
        """Updates main critics every step, update main actor and all targets every d steps."""
        self.step_counter+=1

        # wait until we have a full batch of size N
        if self.buffer.get_buffer_size()<self.N: 
            return
        # sample mini batch of transitions
        buffer_sample=self.buffer.get_random_sample(self.N)
        state_list,action_list, reward_list, new_state_list, terminated_list, truncated_list=zip(*buffer_sample)

        # convert sampled transition to pytorch tensors
        state_batch=torch.tensor(np.array(state_list),dtype=torch.float32)
        action_batch=torch.tensor(np.array(action_list),dtype=torch.float32)
        reward_batch=torch.tensor(np.array(reward_list),dtype=torch.float32)
        new_state_batch=torch.tensor(np.array(new_state_list),dtype=torch.float32)
        terminated_batch=torch.tensor(np.array(terminated_list),dtype=torch.float32)

        # Get target Q-value using target actors and target critics
        with torch.no_grad():
            action_tilda=self.actor_target(new_state_batch)
            epsilon=torch.clamp(torch.randn_like(action_tilda)*self.sigma_tilda, -self.c, self.c)

            # new epsilon every time
            action_tilda+=epsilon
            action_tilda=torch.clamp(action_tilda,-self.action_magnitude,self.action_magnitude)

            q_1_target=self.critic_1_target(torch.cat((new_state_batch,action_tilda),dim=1))
            q_2_target=self.critic_2_target(torch.cat((new_state_batch,action_tilda),dim=1))

            # choose smaller target q value to reduce overestimation
            y= (
                torch.reshape(reward_batch,(self.N,1))
                + self.gamma
                *(1-torch.reshape(terminated_batch,(self.N,1)))
                *torch.minimum(q_1_target,q_2_target)
            )

        # Update main critic 1 by minimizing MSE between predicted q val and target q vals
        self.critic_1_optimizer.zero_grad()
        outputs_main_critic_1=self.critic_1_main(torch.cat((state_batch,action_batch),dim=1))
        loss_critic_1=self.main_critic_loss_fn(outputs_main_critic_1, y)
        self.critic_1_loss.append([self.step_counter,loss_critic_1.item()])
        loss_critic_1.backward()
        self.critic_1_optimizer.step()

        # Update main critic 2 by minimizing MSE between predicted q val and target q vals
        self.critic_2_optimizer.zero_grad()
        outputs_main_critic_2=self.critic_2_main(torch.cat((state_batch,action_batch),dim=1))
        loss_critic_2=self.main_critic_loss_fn(outputs_main_critic_2, y)
        self.critic_2_loss.append([self.step_counter,loss_critic_2.item()])
        loss_critic_2.backward()
        self.critic_2_optimizer.step()

        # Update main actor and target critics every d steps 
        if self.step_counter%self.d==0:
            self.main_actor_optimizer.zero_grad()
            # Update main actor to maximize critic 1s q-value for actor's selected action
            main_actor_actions=self.actor_main(state_batch)
            outputs_main_critic_1=self.critic_1_main(torch.cat((state_batch, main_actor_actions),dim=1))
            # why negative? Adam optimizer performs minimization
            # therfore wnat to minimize teh negative q value instead, which is postive
            loss_actor= outputs_main_critic_1.mean()*-1
            self.actor_loss.append([self.step_counter,loss_actor.item()])
            loss_actor.backward()
            self.main_actor_optimizer.step()

            # update target networks
            self.update_target_networks()

    def update_target_networks(self):
        """Update target network params toward main networks"""
        with torch.no_grad():
            for actor_main_params, actor_target_params in zip(self.actor_main.parameters(), self.actor_target.parameters()):
                actor_target_params.mul_(1-self.tau)
                actor_target_params.add_(self.tau*actor_main_params)

            for critic1_main_params, critic1_target_params in zip(self.critic_1_main.parameters(), self.critic_1_target.parameters()):
                critic1_target_params.mul_(1-self.tau)
                critic1_target_params.add_(self.tau*critic1_main_params)

            for critic2_main_params, critic2_target_params in zip(self.critic_2_main.parameters(), self.critic_2_target.parameters()):
                critic2_target_params.mul_(1-self.tau)
                critic2_target_params.add_(self.tau*critic2_main_params)



