"""Test script for PyTorch model parameters.

This script demonstrates how to inspect model parameters
and count the total number of parameters in a neural network.
"""
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Simple feedforward neural network for testing.
    
    Attributes:
        linear1: First linear layer (10 -> 20).
        linear2: Second linear layer (20 -> 1).
    """
    
    def __init__(self) -> None:
        """Initialize the simple model."""
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 10).
            
        Returns:
            Output tensor of shape (batch_size, 1).
        """
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def main() -> None:
    """Print model parameters information."""
    model = SimpleModel()

    print("Model Parameters:")
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of parameters: {total_params}")


if __name__ == "__main__":
    main()