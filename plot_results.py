import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os



def plot_results():
    """
    Load saved TD3 training metrics and generate plots
    """
    os.makedirs("results/plots", exist_ok=True)
    training_rewards=np.load('results/training_rewards.npy')
    eval_rewards=np.load('results/evaluation_rewards.npy')
    actor_losses=np.load('results/actor_loss.npy')
    critic_1_losses=np.load('results/critic_1_loss.npy')
    critic_2_losses=np.load('results/critic_2_loss.npy')

    # Plot 1: training episode return with 10 ep rolling mean
    plt.plot(pd.Series(training_rewards).rolling(window=10).mean())
    plt.xlabel('Episode')
    plt.ylabel('Episode Return (10 Episode Rolling Mean)')
    plt.title('TD3 PendV1: Training Return')
    plt.savefig("results/plots/training_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Mean evaluation episode reutnr every 5000 observations
    plt.plot(np.array(eval_rewards)[:,0]
    ,np.array(eval_rewards)[:,1])
    plt.xlabel('Environment Steps')
    plt.ylabel('Mean Evaluation Return (5 Episodes)')
    plt.title('TD3 PendV1: Evaluation Return (mean of 5 episode returns, every 5000 steps)')
    plt.savefig("results/plots/eval_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Critic 1 Loss with 50 update rolling average 
    plt.plot(critic_1_losses[:,0],pd.Series(critic_1_losses[:,1]).rolling(window=50).mean())
    plt.xlabel('Environment Steps')
    plt.ylabel('Critic 1 Loss (50 step Rolling Mean)')
    plt.title('TD3 PendV1: Critic 1 Loss')
    plt.savefig("results/plots/critic_1_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Critic 2 Loss with 50 update rolling average 
    plt.plot(critic_2_losses[:,0], pd.Series(critic_2_losses[:,1]).rolling(window=50).mean())
    plt.xlabel('Environment Steps')
    plt.ylabel('Critic 2 Loss (50 step Rolling Mean)')
    plt.title('TD3 PendV1: Critic 2 Loss')
    plt.savefig("results/plots/critic_2_loss.png", dpi=300, bbox_inches="tight")
    plt.close()


    # Plot 5: Raw actor loss at each actor update
    plt.plot(actor_losses[:,0]
    ,actor_losses[:,1])
    plt.xlabel('Environment Steps')
    plt.ylabel('Actor Loss (Raw)')
    plt.title('TD3 PendV1: Actor Loss (−mean Q₁(s, π(s))) per Actor Update')
    plt.savefig("results/plots/actor_losses_raw.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 6: actor loss w/ 100-update rolling average
    plt.plot(actor_losses[:,0],pd.Series(actor_losses[:,1]).rolling(window=100).mean())
    plt.xlabel('Environment Steps')
    plt.ylabel('Actor Loss 100 Update Rolling Mean')
    plt.title('TD3 PendV1: Actor Loss (−mean Q₁(s, π(s))), 100 Update Rolling Mean')
    plt.savefig("results/plots/actor_losses_rolling_avg.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__=='__main__':
    
    plot_results()