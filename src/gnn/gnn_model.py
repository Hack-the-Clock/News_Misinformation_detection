"""Graph Neural Network model for fact verification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data


class FactVerificationGNN(nn.Module):
    """
    Graph Neural Network for fact verification and contradiction detection
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Initialize GNN model

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_attention: Use Graph Attention Network vs Graph Convolution
        """
        super(FactVerificationGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Graph convolution layers
        self.convs = nn.ModuleList()

        if use_attention:
            # Graph Attention Network
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            final_dim = hidden_dim * 4
        else:
            # Graph Convolutional Network
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            final_dim = hidden_dim

        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * 4 if use_attention else hidden_dim)
            for _ in range(num_layers)
        ])

        # Output layers
        self.fc1 = nn.Linear(final_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data):
        """
        Forward pass

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Node-level predictions
        """
        x, edge_index = data.x, data.edge_index

        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layers
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def get_node_embeddings(self, data: Data):
        """
        Get node embeddings (before final classification layer)

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Node embeddings
        """
        x, edge_index = data.x, data.edge_index

        # Graph convolution layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # First FC layer for embeddings
        x = self.fc1(x)
        x = F.relu(x)

        return x


class ContradictionScorer(nn.Module):
    """
    Neural network to score contradiction likelihood between node pairs
    """

    def __init__(self, embedding_dim: int = 64):
        """
        Initialize contradiction scorer

        Args:
            embedding_dim: Dimension of node embeddings
        """
        super(ContradictionScorer, self).__init__()

        # Pairwise scoring network
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor):
        """
        Score contradiction between two nodes

        Args:
            embedding1: First node embedding
            embedding2: Second node embedding

        Returns:
            Contradiction score (0-1)
        """
        # Concatenate embeddings
        combined = torch.cat([embedding1, embedding2], dim=-1)

        # Forward pass
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))

        return x


class NeurosymbolicVerifier:
    """
    Combines GNN predictions with symbolic logic rules
    """

    def __init__(self, gnn_model: FactVerificationGNN, symbolic_weight: float = 0.5):
        """
        Initialize verifier

        Args:
            gnn_model: Trained GNN model
            symbolic_weight: Weight for symbolic rules (0-1), remainder is GNN weight
        """
        self.gnn_model = gnn_model
        self.symbolic_weight = symbolic_weight
        self.gnn_weight = 1.0 - symbolic_weight

    def verify_facts(
        self,
        data: Data,
        symbolic_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine GNN and symbolic scores

        Args:
            data: Graph data
            symbolic_scores: Scores from symbolic logic rules

        Returns:
            Combined credibility scores
        """
        # Get GNN predictions
        self.gnn_model.eval()
        with torch.no_grad():
            gnn_output = self.gnn_model(data)
            gnn_scores = torch.exp(gnn_output[:, 1])  # Probability of class 1 (credible)

        # Combine scores
        combined_scores = (
            self.symbolic_weight * symbolic_scores +
            self.gnn_weight * gnn_scores
        )

        return combined_scores

    def set_weights(self, symbolic_weight: float):
        """Update symbolic vs GNN weight"""
        self.symbolic_weight = symbolic_weight
        self.gnn_weight = 1.0 - symbolic_weight
