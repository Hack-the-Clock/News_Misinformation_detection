"""Convert NetworkX graph to PyTorch Geometric format"""

import torch
import networkx as nx
from torch_geometric.data import Data
from typing import Dict, List
import numpy as np


class GraphEmbedding:
    """Convert temporal knowledge graph to PyTorch Geometric format"""

    def __init__(self, feature_dim: int = 32):
        """
        Initialize graph embedding

        Args:
            feature_dim: Dimension of node features
        """
        self.feature_dim = feature_dim

    def networkx_to_pyg(self, nx_graph: nx.Graph) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object

        Args:
            nx_graph: NetworkX graph

        Returns:
            PyTorch Geometric Data object
        """
        # Create node mapping
        node_list = list(nx_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Extract node features
        node_features = []
        node_labels = []

        for node in node_list:
            node_data = nx_graph.nodes[node]
            features = self._extract_node_features(node_data)
            node_features.append(features)

            # Label: 0 for entity, 1 for event
            label = 1 if node_data.get('node_type') == 'event' else 0
            node_labels.append(label)

        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        # Extract edges
        edge_index = []
        edge_attr = []

        for source, target, edge_data in nx_graph.edges(data=True):
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]

            edge_index.append([source_idx, target_idx])

            # Extract edge features
            edge_features = self._extract_edge_features(edge_data)
            edge_attr.append(edge_features)

        # Convert to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        if edge_attr:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_attr = None

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=len(node_list)
        )

        return data

    def _extract_node_features(self, node_data: Dict) -> List[float]:
        """
        Extract numerical features from node data

        Args:
            node_data: Node attributes

        Returns:
            Feature vector
        """
        features = []

        # Node type (one-hot)
        node_type = node_data.get('node_type', 'entity')
        features.append(1.0 if node_type == 'entity' else 0.0)
        features.append(1.0 if node_type == 'event' else 0.0)

        # Confidence score
        confidence = node_data.get('confidence', 0.5)
        features.append(confidence)

        # Has timestamp (binary)
        has_timestamp = 1.0 if node_data.get('timestamp') else 0.0
        features.append(has_timestamp)

        # Timestamp encoding (if available)
        if node_data.get('timestamp'):
            timestamp = node_data['timestamp']
            # Encode as day of year and hour (normalized)
            day_of_year = timestamp.timetuple().tm_yday / 365.0
            hour = timestamp.hour / 24.0
            features.extend([day_of_year, hour])
        else:
            features.extend([0.0, 0.0])

        # Entity type encoding (for entities)
        if node_type == 'entity':
            entity_type = node_data.get('entity_type', 'concept')
            entity_types = ['location', 'organization', 'person', 'source', 'concept']
            for etype in entity_types:
                features.append(1.0 if entity_type == etype else 0.0)
        else:
            features.extend([0.0] * 5)

        # Pad or truncate to feature_dim
        while len(features) < self.feature_dim:
            features.append(0.0)

        return features[:self.feature_dim]

    def _extract_edge_features(self, edge_data: Dict) -> List[float]:
        """
        Extract numerical features from edge data

        Args:
            edge_data: Edge attributes

        Returns:
            Feature vector
        """
        features = []

        # Confidence
        confidence = edge_data.get('confidence', 0.5)
        features.append(confidence)

        # Relation type encoding
        relation_type = edge_data.get('relation_type', 'involves')
        relation_types = [
            'reported', 'confirmed', 'contradicts', 'supports',
            'causes', 'happens_at', 'involves', 'temporal_before', 'temporal_after'
        ]

        for rtype in relation_types:
            features.append(1.0 if relation_type == rtype else 0.0)

        # Has timestamp
        has_timestamp = 1.0 if edge_data.get('timestamp') else 0.0
        features.append(has_timestamp)

        return features

    def add_contradiction_labels(
        self,
        pyg_data: Data,
        contradictions: List,
        node_list: List
    ) -> Data:
        """
        Add contradiction information as node/edge labels

        Args:
            pyg_data: PyTorch Geometric data
            contradictions: List of contradictions
            node_list: List of nodes in order

        Returns:
            Updated data object
        """
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Create contradiction matrix
        num_nodes = pyg_data.num_nodes
        contradiction_matrix = torch.zeros((num_nodes, num_nodes))

        for contradiction in contradictions:
            id1 = contradiction.fact1_id
            id2 = contradiction.fact2_id

            if id1 in node_to_idx and id2 in node_to_idx:
                idx1 = node_to_idx[id1]
                idx2 = node_to_idx[id2]

                contradiction_matrix[idx1, idx2] = contradiction.severity
                contradiction_matrix[idx2, idx1] = contradiction.severity

        pyg_data.contradiction_matrix = contradiction_matrix

        return pyg_data
