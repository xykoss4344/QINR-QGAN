import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import os
import argparse

def plot_loss(csv_path="results_crystal_qgan/training_loss_history.csv", save_path="loss_curve.png"):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        print("Make sure you have trained the model using train_crystal.py to generate this file.")
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['d_loss'], label='Critic Loss', color='red')
        plt.plot(df['epoch'], df['g_wgan_loss'], label='Generator WGAN Loss', color='blue')
        plt.plot(df['epoch'], df['physics_loss'], label='Physics Penalty', color='green')
        plt.plot(df['epoch'], df['total_g_loss'], label='Total Generator Loss', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('QINR-QGAN Training Loss Over Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Loss curve successfully saved to {save_path}")
        plt.close()
        
    except ImportError:
        print("Pandas is not installed. Plotting using raw csv parser...")
        import csv
        epochs, d_losses, g_wgan, physics, total_g = [], [], [], [], []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                d_losses.append(float(row['d_loss']))
                g_wgan.append(float(row['g_wgan_loss']))
                physics.append(float(row['physics_loss']))
                total_g.append(float(row['total_g_loss']))
                
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, d_losses, label='Critic Loss', color='red')
        plt.plot(epochs, g_wgan, label='Generator WGAN Loss', color='blue')
        plt.plot(epochs, physics, label='Physics Penalty', color='green')
        plt.plot(epochs, total_g, label='Total Generator Loss', color='purple')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('QINR-QGAN Training Loss Over Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Loss curve successfully saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QINR-QGAN training loss")
    parser.add_argument("--csv_path", type=str, default="results_crystal_qgan/training_loss_history.csv")
    parser.add_argument("--save_path", type=str, default="loss_curve.png")
    args = parser.parse_args()
    
    plot_loss(args.csv_path, args.save_path)
