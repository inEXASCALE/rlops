
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_progress(results_dir='results_random_unsym'):
    """
    Plot training progress (average episode rewards and TD loss) from saved data.
    
    Parameters:
    - results_dir: Directory where episode_rewards.npy and training_losses.npy are saved.
    """
    # Load data
    rewards_path = os.path.join(results_dir, 'episode_rewards.npy')
    losses_path = os.path.join(results_dir, 'training_losses.npy')
    
    if not os.path.exists(rewards_path):
        print(f"Error: Rewards file not found in {rewards_path}")
        return
    
    episode_rewards = np.load(rewards_path)
    
    # Plot average episode rewards
    plt.figure(figsize=(7.5, 5))
    plt.plot(range(len(episode_rewards)), episode_rewards, label='Average Episode Reward', color='blue')
    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Average Reward', fontsize=15)
    plt.title('Average Reward per Episode', fontsize=15)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'episode_rewards_plot_dense3.png'))
    plt.close()
    print(f"Saved episode rewards plot to {os.path.join(results_dir, 'episode_rewards_plot_dense3.png')}")

    # Plot training losses if available
    if os.path.exists(losses_path):
        training_losses = np.load(losses_path)
        plt.figure(figsize=(7.5, 5))
        plt.plot(range(len(training_losses)), training_losses, label='Average RPE', color='red')
        plt.xlabel('Episode', fontsize=15)
        plt.ylabel('RPE', fontsize=15)
        plt.title('Average RPE per Episode', fontsize=15)
        plt.legend(fontsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'training_losses_plot_dense3.png'))
        plt.close()
        print(f"Saved training losses plot to {os.path.join(results_dir, 'training_losses_plot_dense3.png')}")
    else:
        print(f"Warning: Training losses file not found in {losses_path}. Skipping loss plot.")



if __name__ == "__main__":
    results_dir = 'results_random_dense3'
    plot_training_progress(results_dir)