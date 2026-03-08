
import os
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.autograd as autograd
from models.QINR_Crystal import PQWGAN_CC_Crystal

def compute_gradient_penalty(dict_D, real_samples, fake_samples, labels, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Conditional Critic takes data + labels
    # We concat in the forward pass of Critic or here?
    # Our ClassicalCritic takes 'input_dim' which is data_dim + label_dim
    # So we need to concat before passing
    interpolates_cond = torch.cat([interpolates, labels], dim=1)
    d_interpolates = dict_D(interpolates_cond)
    
    fake = torch.ones(real_samples.shape[0], 1).to(device).requires_grad_(False)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_distance_penalty(coords_batch, min_dist=1.0):
    """
    Computes a physics-informed penalty for interatomic distances < min_dist.
    coords_batch: shape (batch_size, 90) representing (batch_size, 30 atoms, 3 coords)
    """
    batch_size = coords_batch.shape[0]
    # Reshape to (batch, 30 atoms, 3 xyz coords)
    coords = coords_batch.view(batch_size, 30, 3)
    
    # Calculate pairwise Euclidean distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1) # shape (batch, 30, 30, 3)
    # Add epsilon to prevent NaN gradients from sqrt(0)
    distances = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8) # shape (batch, 30, 30)
    
    # Mask out diagonal (distance from an atom to itself)
    mask = torch.eye(30, device=coords.device).unsqueeze(0).expand(batch_size, -1, -1).bool()
    distances_masked = distances.masked_fill(mask, float('inf'))
    
    # Violations: any distance less than min_dist
    violations = torch.relu(min_dist - distances_masked)
    
    # Sum violations across atoms, mean across the batch
    penalty = torch.mean(torch.sum(violations, dim=(1,2)))
    return penalty

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    print(f"Loading dataset from {args.dataset_path}")
    with open(args.dataset_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Format: List of (coords, labels)
    # coords: (30, 3) -> flatten to 90
    # labels: (28, 1) -> flatten to 28
    
    train_data_coords = []
    train_data_labels = []
    
    for c, l in raw_data:
        train_data_coords.append(c.flatten())
        train_data_labels.append(l.flatten())
        
    train_data_coords = np.array(train_data_coords, dtype=np.float32)
    train_data_labels = np.array(train_data_labels, dtype=np.float32)
    
    # Normalize coords? They look like fractional coordinates [0,1], but some are 0.
    # WGAN usually works better with [-1, 1] or [0, 1].
    # Let's assume they are in [0, 1].
    
    dataset = TensorDataset(torch.from_numpy(train_data_coords), torch.from_numpy(train_data_labels))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    data_dim = 90
    label_dim = 28
    z_dim = args.z_dim
    hidden_features = args.hidden_features # Qubits
    hidden_layers = args.hidden_layers
    spectrum = args.spectrum_layer
    use_noise = args.use_noise
    
    # Generator Input: z + label
    gen_input_dim = z_dim + label_dim
    
    # Critic Input: data + label
    critic_input_dim = data_dim + label_dim
    
    print(f"Initializing QINR Crystal Model...")
    gan = PQWGAN_CC_Crystal(
        input_dim_g=gen_input_dim,
        output_dim=data_dim,
        input_dim_d=critic_input_dim,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        spectrum_layer=spectrum,
        use_noise=use_noise
    )
    
    generator = gan.generator.to(device)
    critic = gan.critic.to(device)
    
    lr_G = args.lr_g
    lr_D = args.lr_d
    b1 = 0.0
    b2 = 0.9
    
    optimizer_G = Adam(generator.parameters(), lr=lr_G, betas=(b1, b2))
    optimizer_C = Adam(critic.parameters(), lr=lr_D, betas=(b1, b2))
    
    lambda_gp = 10
    n_critic = 5
    
    for epoch in range(args.n_epochs):
        for i, (real_imgs, labels) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            current_batch_size = real_imgs.shape[0]
            
            # --- Train Critic ---
            optimizer_C.zero_grad()
            
            # Sample noise
            z = torch.randn(current_batch_size, z_dim).to(device)
            
            # Generate fake images (Conditioned)
            gen_input = torch.cat([z, labels], dim=1)
            fake_imgs = generator(gen_input)
            
            # Critic score for Real (Conditioned)
            critic_input_real = torch.cat([real_imgs, labels], dim=1)
            real_validity = critic(critic_input_real)
            
            # Critic score for Fake (Conditioned)
            critic_input_fake = torch.cat([fake_imgs, labels], dim=1)
            fake_validity = critic(critic_input_fake)
            

            # Gradient Penalty
            gradient_penalty = compute_gradient_penalty(critic, real_imgs, fake_imgs, labels, device)

            
            # Adversarial Loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            
            d_loss.backward()
            optimizer_C.step()
            
            # --- Train Generator ---
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                
                # Generate fake images
                z = torch.randn(current_batch_size, z_dim).to(device)
                gen_input = torch.cat([z, labels], dim=1)
                fake_imgs = generator(gen_input)
                
                # Score with Critic
                critic_input_fake = torch.cat([fake_imgs, labels], dim=1)
                fake_validity = critic(critic_input_fake)
                
                g_wgan_loss = -torch.mean(fake_validity)
                
                # Physics-Informed Loss
                physics_loss = compute_distance_penalty(fake_imgs, min_dist=args.min_dist)
                
                # Total Generator Loss
                g_loss = g_wgan_loss + (args.lambda_physics * physics_loss)
                
                g_loss.backward()
                optimizer_G.step()
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G WGAN: {g_wgan_loss.item():.4f}] [G Physics: {physics_loss.item():.4f}]")

        # Save checkopints
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.out_folder, f"checkpoint_{epoch}.pt")
            torch.save({
                'generator': generator.state_dict(),
                'critic': critic.state_dict(),
                'epoch': epoch
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=r"c:\Users\Jeremy\Documents\Projects\qgan\Composition-Conditioned-Crystal-GAN\Composition_Conditioned_Crystal_GAN\preparing_dataset\mgmno_100.pickle")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--z_dim", type=int, default=16)
    parser.add_argument("--hidden_features", type=int, default=6) # Qubits
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--spectrum_layer", type=int, default=2)
    parser.add_argument("--use_noise", type=float, default=0.0)
    parser.add_argument("--lr_g", type=float, default=0.0001)
    parser.add_argument("--lr_d", type=float, default=0.0001)
    parser.add_argument("--out_folder", type=str, default="./results_crystal_qgan")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--lambda_physics", type=float, default=5.0, help="Weight of the interatomic distance penalty")
    parser.add_argument("--min_dist", type=float, default=1.0, help="Minimum accepted distance between atoms")
    
    args = parser.parse_args()
    
    os.makedirs(args.out_folder, exist_ok=True)
    train(args)
