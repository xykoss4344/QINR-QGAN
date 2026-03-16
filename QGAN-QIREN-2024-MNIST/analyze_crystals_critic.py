import os
import sys
import pickle
import torch
import numpy as np
import random
import glob
import matplotlib.pyplot as plt

from models.QINR_Crystal import PQWGAN_CC_Crystal
original_repo_path = r"c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\datasets"
sys.path.append(original_repo_path)
from view_atoms_mgmno import view_atoms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset (Real Crystals)
    dataset_path = "datasets/mgmno_100.pickle"
    with open(dataset_path, 'rb') as f:
        raw_data = pickle.load(f)
        
    num_samples = 1000
    real_coords_list = []
    labels_list = []
    
    # Randomly sample 1000 real crystals with their labels
    # raw_data has length 100, so we sample with replacement
    for _ in range(num_samples):
        c, l = random.choice(raw_data)
        real_coords_list.append(c.flatten())
        labels_list.append(l.flatten())
        
    real_coords = torch.tensor(np.array(real_coords_list, dtype=np.float32)).to(device)
    labels = torch.tensor(np.array(labels_list, dtype=np.float32)).to(device)
    
    # 2. Init Model
    z_dim = 16
    label_dim = 28
    data_dim = 90
    gen_input_dim = z_dim + label_dim
    critic_input_dim = data_dim + label_dim
    
    gan = PQWGAN_CC_Crystal(
        input_dim_g=gen_input_dim,
        output_dim=data_dim,
        input_dim_d=critic_input_dim,
        hidden_features=6,
        hidden_layers=2,
        spectrum_layer=2,
        use_noise=0.0
    )
    generator = gan.generator.to(device)
    critic = gan.critic.to(device)
    
    # Load checkpoint
    checkpoint_path = "./results_crystal_qgan/checkpoint_90.pt"
    if not os.path.exists(checkpoint_path):
        ckpts = glob.glob("./results_crystal_qgan/checkpoint_*.pt")
        if ckpts:
            checkpoint_path = sorted(ckpts, key=os.path.getmtime)[-1]
        else:
            print("No checkpoints found!")
            return

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    critic.load_state_dict(checkpoint['critic'])
    
    generator.eval()
    critic.eval()
    
    # 3. Generate Fake Crystals
    print(f"Generating {num_samples} fake crystal structures...")
    batch_size = 100
    fake_coords_list = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_labels = labels[i:i+batch_size]
            z = torch.randn(batch_labels.shape[0], z_dim).to(device)
            gen_input = torch.cat([z, batch_labels], dim=1)
            fake_imgs = generator(gen_input)
            fake_coords_list.append(fake_imgs)
            
    fake_coords = torch.cat(fake_coords_list, dim=0)
    
    # 4. Evaluate with Critic
    print("Evaluating with Critic network...")
    real_scores_list = []
    fake_scores_list = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Real
            b_real = real_coords[i:i+batch_size]
            b_labels = labels[i:i+batch_size]
            critic_input_real = torch.cat([b_real, b_labels], dim=1)
            real_validity = critic(critic_input_real).cpu().numpy().flatten()
            real_scores_list.extend(real_validity)
            
            # Fake
            b_fake = fake_coords[i:i+batch_size]
            critic_input_fake = torch.cat([b_fake, b_labels], dim=1)
            fake_validity = critic(critic_input_fake).cpu().numpy().flatten()
            fake_scores_list.extend(fake_validity)
            
    real_scores = np.array(real_scores_list)
    fake_scores = np.array(fake_scores_list)
    
    # Calculate Statistics
    real_mean, real_std = np.mean(real_scores), np.std(real_scores)
    fake_mean, fake_std = np.mean(fake_scores), np.std(fake_scores)
    
    # Wasserstein distance approximation E[D(x)] - E[D(G(z))]
    wasserstein_dist = real_mean - fake_mean
    
    print("\n===========================================")
    print("      CRITIC EVALUATION RESULTS TABLE     ")
    print("===========================================")
    print(f"{'Metric':<20} | {'Real Crystals':<15} | {'Generated Crystals':<15}")
    print("-" * 56)
    print(f"{'Mean Critic Score':<20} | {real_mean:>15.5f} | {fake_mean:>15.5f}")
    print(f"{'Std Deviation':<20} | {real_std:>15.5f} | {fake_std:>15.5f}")
    print(f"{'Min Score':<20} | {np.min(real_scores):>15.5f} | {np.min(fake_scores):>15.5f}")
    print(f"{'Max Score':<20} | {np.max(real_scores):>15.5f} | {np.max(fake_scores):>15.5f}")
    print("-" * 56)
    print(f"Wasserstein Distance (Real Mean - Fake Mean): {wasserstein_dist:.5f}")
    print("===========================================\n")
    
    # 5. Plot Histogram
    print("Creating histogram plot...")
    plt.figure(figsize=(10, 6))
    
    # We want a plot like the image: "Relative output of Critic network"
    # To match the relative output, we can center it around 0 by subtracting the real_mean if desired,
    # or just plot the raw scores and add a vertical line at 0.0
    
    plt.hist(fake_scores, bins=30, alpha=0.5, color='sandybrown', edgecolor='darkorange', label='Generated (Fake)')
    # plt.hist(real_scores, bins=30, alpha=0.5, color='steelblue', edgecolor='darkblue', label='Dataset (Real)')
    
    plt.axvline(x=0.0, color='red', linestyle='-', linewidth=2)
    plt.title("QGAN Generated Crystals - Critic Output Distribution")
    plt.xlabel("Output of Critic network")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    
    plot_path = "critic_evaluation_histogram.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Saved histogram to {plot_path}")

if __name__ == '__main__':
    main()
