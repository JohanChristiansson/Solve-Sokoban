import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from constants import RoomState

class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connection = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += skip_connection
        out = F.relu(out)
        return out

class AlphaZero(nn.Module):
    """
    Number of residual blocks have been 20 or 40 when solving Go
    Action size is the number of possible actions, 4 move actions, 4 push actions, and no operation. No operation exists but is should not be used.
    Can try different numbers of input channels.
    """
    def __init__(
        self,
        board_size: int,
        embedding_dim: int,
        lr: float,
        betas: tuple,
        weight_decay: float,
        num_residual_blocks: int=39,
        num_channels: int=256,
        action_size: int=4,
    ):
        super(AlphaZero, self).__init__()

        # Extract the dimensions of the board
        board_x, board_y = board_size

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=len(RoomState), embedding_dim=embedding_dim)
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(embedding_dim, num_channels, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(num_channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([ResidualBlock(num_channels) for _ in range(num_residual_blocks)])

        # Policy head - Temporarily we do not have a policy head
        #self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)  # Reduce channels to 2
        #self.policy_bn = nn.BatchNorm2d(2)
        #self.policy_fc = nn.Linear(2 * board_size * board_size, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)  # Reduce channels to 1
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_x * board_y + 1, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Create an optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x: Union[np.ndarray, torch.Tensor], heuristics: np.ndarray) -> tuple[Union[torch.Tensor, None], torch.Tensor]:
        # Convert x to a tensor if it isn't already
        if not isinstance(x, torch.Tensor):
            board_tensor = torch.tensor(x, dtype=torch.long, device=self.device)
        else:
            board_tensor = x.clone().long()

        heuristics = torch.tensor(heuristics, dtype=torch.float32, device=self.device)

        # If the batch and channel dimensions are not present, add them
        if board_tensor.dim() == 2:
            board_tensor = board_tensor.unsqueeze(0)

        # Pass through the embedding layer
        board_tensor = self.embedding(board_tensor)

        # Current shape is (batch_size, board_x, board_y, embedding_dim), reshape to (batch_size, embedding_dim, board_x, board_y)
        board_tensor = board_tensor.permute(0, 3, 1, 2)

        # Initial convolutional block
        board_tensor = F.relu(self.bn1(self.conv1(board_tensor)))

        # Pass through residual blocks
        for block in self.residual_blocks:
            board_tensor = block(board_tensor)

        # Policy head
        #policy = F.relu(self.policy_bn(self.policy_conv(board_tensor)))
        #policy = policy.view(policy.size(0), -1)  # Flatten
        #policy = self.policy_fc(policy)
        #policy = F.log_softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(board_tensor)))
        value = value.view(value.size(0), -1)  # Flatten
        heuristics = heuristics.view(-1, 1)
        value = torch.cat((value, heuristics), dim= -1)
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)  # No activation function, so linear

        return None, value
    
    def train_model(self, x: np.ndarray, target_value: np.ndarray, heuristics: np.ndarray):
        # Convert x and target_value to tensors
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        target_value = torch.tensor(target_value, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        self.train() # Set the model to training mode
        criterion_value = nn.MSELoss() # Loss for value head

        self.optimizer.zero_grad()

        # Forward pass
        _, predicted_value = self.forward(x, heuristics)

        # Calculate loss
        loss_value = criterion_value(predicted_value, target_value)

        # Backward pass
        loss_value.backward()
        self.optimizer.step()

        print(f'Loss (Value): {loss_value.item():.4f}')
        
        self.eval()
