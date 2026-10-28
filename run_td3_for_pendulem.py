
from replay_buffer import replay_buffer
from actor_critic_networks import ActorNetwork, CriticNetwork
from td3 import TD3
import gymnasium as gym
import numpy as np
import os
from plot_results import plot_results
import random
import torch

def evalulation_helper(step,td3_agent,env_eval,total_eval_episodes=5):
    """
    Returns: [environment step, mean episode return across evalution episodes]
    """
    tmp=[]
    eval_reward_sum=0
    eval_state, info = env_eval.reset() 
    eval_episode_counter=0

    # run evalution episode using deterministic policy
    while eval_episode_counter<total_eval_episodes:
        action_eval=td3_agent.select_action(eval_state,explore=False)
        eval_new_state, eval_reward, eval_terminated,eval_truncated, eval_info= env_eval.step(action_eval)
        eval_state=eval_new_state
        eval_reward_sum+=eval_reward
        if eval_terminated or eval_truncated:
            eval_episode_counter+=1
            tmp.append(eval_reward_sum)
            eval_reward_sum=0
            eval_state, info = env_eval.reset() 
    # Return mean evalutation return at this training step 
    return [step,np.mean(tmp)]


def main():
    # create separate environments for training and evaluation
    seed=42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_train = gym.make("Pendulum-v1")
    env_eval=gym.make("Pendulum-v1")

    reward_tracker=[]
    reward_sum=0
    max_steps=50000
    eval_reward_tracker=[]

    td3_agent=TD3()
    state, info = env_train.reset(seed=seed)
    env_eval.reset(seed=seed+100)



    for step in range(max_steps): 
        # Select an action and observe the transtions
        action=td3_agent.select_action(state,explore=True)
        new_state, reward, terminated,truncated, info= env_train.step(action)
        # store transition and accumalate episode return
        td3_agent.store_in_buffer(state, action, reward,new_state,terminated, truncated)
        reward_sum+=reward
        state=new_state

        # evalutate the policy without exploration noise
        if step%5000==0:
            eval_reward_tracker.append(evalulation_helper(step, td3_agent,env_eval,total_eval_episodes=5))        

        # Record completed episode returns and reset environmnet 
        if terminated or truncated: 
            reward_tracker.append(reward_sum)
            reward_sum=0 
            state,info=env_train.reset()

        # Update the td3 agent using the replay buffer mini batch
        td3_agent.train()


    # save training metrics and generate plots
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    np.save("results/training_rewards.npy",np.array(reward_tracker))
    np.save("results/evaluation_rewards.npy",np.array(eval_reward_tracker))
    np.save("results/actor_loss.npy",np.array(td3_agent.actor_loss))
    np.save("results/critic_1_loss.npy",np.array(td3_agent.critic_1_loss))
    np.save("results/critic_2_loss.npy",np.array(td3_agent.critic_2_loss))
    plot_results()

    env_train.close()
    env_eval.close()

if __name__=='__main__':
    main()