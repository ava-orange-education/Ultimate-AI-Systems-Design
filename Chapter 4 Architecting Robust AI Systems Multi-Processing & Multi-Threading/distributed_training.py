import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import os
import time
import argparse
import warnings
import GPUtil
import psutil

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

"""
Setup Instructions:
1. Ensure PyTorch is installed with CUDA support: pip install torch torchvision torchaudio
2. Install additional dependencies: pip install psutil gputil
3. Make sure you have multiple GPUs available for distributed training
4. Run the script with: python script_name.py --world_size 2 (or the number of GPUs you want to use)
"""

class ComplexCNN(nn.Module):
    """
    A complex CNN model for regression tasks.
    
    This model consists of multiple 1D convolutional layers followed by fully connected layers.
    It includes batch normalization, dropout, and LeakyReLU activation for improved training dynamics.
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ComplexCNN, self).__init__()
        # Convolutional layers
        self.layers = nn.ModuleList([nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1)])
        self.layers.extend([nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1) for _ in range(num_layers - 2)])
        self.layers.append(nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1))
        
        # Additional layers for complex processing
        self.activation = nn.LeakyReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Apply convolutional layers
        x = x.unsqueeze(1)  # Add channel dimension
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.squeeze(1)  # Remove channel dimension
        
        # Apply fully connected layers with additional processing
        x = self.activation(self.fc1(x))
        x = self.batch_norm(x)
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class ComplexRegressionDataset(Dataset):
    """
    A custom dataset for complex regression tasks.
    
    Generates random input data and calculates target values as the average of input features.
    """
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.data = torch.randn(size, dim)  # Random input data
        self.targets = torch.sum(self.data, dim=1, keepdim=True) / dim  # Target is average of inputs

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

def train(rank, world_size, args):
    """
    Main training function for each process.
    
    Args:
        rank (int): Unique identifier of the process
        world_size (int): Total number of processes
        args (Namespace): Command-line arguments
    """
    setup(rank, world_size)
    torch.manual_seed(42 + rank)  # Set different seeds for each process
    
    # Set up device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # Create and wrap model in DistributedDataParallel
    model = ComplexCNN(args.input_dim, args.hidden_dim, args.num_layers).to(device)
    model = DDP(model, device_ids=[rank])

    # Create dataset and dataloader
    dataset = ComplexRegressionDataset(size=args.dataset_size, dim=args.input_dim)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)

    # Set up optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        epoch_start_time = time.time()
        total_loss = 0
        
        for batch, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs.squeeze(), targets.squeeze())

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

            # Print progress and resource usage
            if batch % 10 == 0:
                gpu = GPUtil.getGPUs()[rank]
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch}/{len(dataloader)}, Loss: {loss.item():.4f}")
                print(f"GPU {rank} Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                print(f"GPU {rank} Utilization: {gpu.load*100:.2f}%")
                print(f"CPU Usage: {psutil.cpu_percent()}%, RAM Usage: {psutil.virtual_memory().percent}%")

        # Print epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Rank {rank}, Epoch {epoch}, Average Loss: {total_loss / len(dataloader):.4f}, Epoch Time: {epoch_time:.2f}s")

    cleanup()

def run_demo(demo_fn, world_size, args):
    """
    Spawn multiple processes to run the demo function.
    
    Args:
        demo_fn (function): The function to run in each process
        world_size (int): Number of processes to spawn
        args (Namespace): Command-line arguments
    """
    mp.spawn(demo_fn,
             args=(world_size, args),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--world_size", type=int, default=3, help="Number of GPUs to use")
    parser.add_argument("--input_dim", type=int, default=2000, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--dataset_size", type=int, default=100000, help="Dataset size")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    args = parser.parse_args()

    # Set CUDA memory allocation to expandable (helps with out of memory errors)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Run the distributed training
    run_demo(train, args.world_size, args)
