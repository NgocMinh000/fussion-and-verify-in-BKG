import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


# reduce dimensions by Autoencoder
class TextEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(TextEmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class BaseRGCN(nn.Module):
    """
    Base class for Relational Graph Convolutional Network (R-GCN) model.
    This class initializes the model and defines the base layers.
    """

    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False, pretrained_text_embeddings=None,
                 pretrained_domain_embeddings=None, freeze=False, w=0.5):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.pretrained_text_embeddings = pretrained_text_embeddings
        self.pretrained_domain_embeddings = pretrained_domain_embeddings
        self.freeze = freeze
        self.w = w

        # Create RGCN layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # Input to hidden layer
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)
        # Hidden to hidden layers
        for idx in range(self.num_hidden_layers):
            hidden_layer = self.build_hidden_layer(idx)
            self.layers.append(hidden_layer)
        # Hidden to output layer (if necessary)
        output_layer = self.build_output_layer()
        if output_layer is not None:
            self.layers.append(output_layer)

    def build_input_layer(self):
        # Override in subclass
        return None

    def build_hidden_layer(self, idx):
        # Override in subclass
        raise NotImplementedError

    def build_output_layer(self):
        # Override in subclass
        return None

    def forward(self, graph, node_ids, rel_ids, norm):
        """
        Forward pass through the RGCN layers.
        """
        for layer in self.layers:
            node_ids = layer(graph, node_ids, rel_ids, norm)
        return node_ids


class EmbeddingLayer(nn.Module):
    """
    Embedding layer to initialize node features with two pretrained embeddings,
    one of which will be linearly transformed to match dimensions, and each is normalized before weighted averaging.
    """

    def __init__(self, num_nodes, hidden_dim, pretrained_text_embeddings, pretrained_domain_embeddings, freeze=False,w=0.5):
        super(EmbeddingLayer, self).__init__()
        self.w = w
        # Pretrained domain embeddings
        if pretrained_domain_embeddings is not None:
            domain_embeddings = torch.from_numpy(pretrained_domain_embeddings).float()
            norm_domain_embeddings = (domain_embeddings - domain_embeddings.min()) / (
                    domain_embeddings.max() - domain_embeddings.min())

            self.poincare_to_euclidean = nn.Linear(pretrained_domain_embeddings.shape[1], hidden_dim)
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(norm_domain_embeddings, freeze=freeze)

            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.poincare_to_euclidean = nn.Linear(hidden_dim, hidden_dim)
            print("Initialized random domain embeddings.")

        # Pretrained text embeddings, which will be linearly transformed
        if pretrained_text_embeddings is not None:
            text_embeddings = torch.from_numpy(pretrained_text_embeddings).float()
            norm_text_embeddings = (text_embeddings - text_embeddings.min()) / (
                    text_embeddings.max() - text_embeddings.min())

            self.norm_text_embeddings = nn.Embedding.from_pretrained(norm_text_embeddings, freeze=freeze)

            self.autoencoder = TextEmbeddingAutoencoder(pretrained_text_embeddings.shape[1], hidden_dim)

            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            self.norm_text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.autoencoder = TextEmbeddingAutoencoder(hidden_dim, hidden_dim)
            print("Initialized random text embeddings.")

    def forward(self, graph, node_ids, rel_ids, norm):
        # Transform the text_embeddings to match the GCN embedding's dimensions
        transformed_text_embeddings, _ = self.autoencoder(self.norm_text_embeddings(node_ids.squeeze()))

        # Map Poincaré embeddings to Euclidean space
        transformed_domain_embeddings = self.poincare_to_euclidean(self.norm_domain_embeddings(node_ids.squeeze()))

        # Weighted average of the two normalized embeddings
        # Assuming equal weight for simplicity; adjust as needed
        combined_embedding = (1 - self.w) * transformed_domain_embeddings + self.w * transformed_text_embeddings

        return combined_embedding


class RGCN(BaseRGCN):
    """
    Implementation of R-GCN with support for link prediction.
    """

    def build_input_layer(self):
        # Initialize node features with embedding layer
        return EmbeddingLayer(self.num_nodes, self.hidden_dim, self.pretrained_text_embeddings,
                              self.pretrained_domain_embeddings, self.freeze, self.w)

    def build_hidden_layer(self, idx):
        # Activation function for all but the last layer
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(in_feat=self.hidden_dim,
                            out_feat=self.hidden_dim,
                            num_rels=self.num_relations,
                            regularizer='bdd',
                            num_bases=self.num_bases,
                            activation=activation,
                            self_loop=self.use_self_loop,
                            dropout=self.dropout)


class LinkPredict(nn.Module):
    """
    Link prediction model using R-GCN.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 pretrained_relation_embeddings=None, freeze=False, w=0.5):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda, pretrained_text_embeddings=pretrained_text_embeddings,
                         pretrained_domain_embeddings=pretrained_domain_embeddings, freeze=freeze,w=w)
        self.regularization_param = regularization_param
        self.hidden_dim = hidden_dim

        # ConvE parameters
        # Reshape embeddings to 2D: hidden_dim should be divisible by embedding_width
        # For hidden_dim=200, we use 10x20
        self.embedding_height = 10
        self.embedding_width = 20
        assert hidden_dim == self.embedding_height * self.embedding_width, \
            f"hidden_dim ({hidden_dim}) must equal embedding_height * embedding_width ({self.embedding_height * self.embedding_width})"

        # Relation embeddings (will be reshaped for ConvE)
        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            normalized_relations = (self.relation_weights - self.relation_weights.min()) / (
                    self.relation_weights.max() - self.relation_weights.min())

            self.relation_weights.data.copy_(normalized_relations)

            print("Loaded pretrained relation embeddings (for ConvE).")
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))
            print("Initialized random relation embeddings (for ConvE).")

        # ConvE CNN components
        self.input_channels = 1
        self.output_channels = 32
        self.kernel_height = 3
        self.kernel_width = 3

        # Batch normalization
        self.bn0 = nn.BatchNorm2d(self.input_channels)
        self.bn1 = nn.BatchNorm2d(self.output_channels)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Dropout layers
        self.input_dropout = nn.Dropout(dropout if dropout > 0 else 0.2)
        self.feature_map_dropout = nn.Dropout2d(dropout if dropout > 0 else 0.2)
        self.output_dropout = nn.Dropout(dropout if dropout > 0 else 0.3)

        # 2D Convolution
        self.conv1 = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=(self.kernel_height, self.kernel_width),
            stride=1,
            padding=0,
            bias=True
        )

        # Calculate flattened feature map size after convolution
        conv_height = 2 * self.embedding_height - self.kernel_height + 1
        conv_width = self.embedding_width - self.kernel_width + 1
        self.flat_size = self.output_channels * conv_height * conv_width

        # Fully connected layer to project back to embedding dimension
        self.fc = nn.Linear(self.flat_size, hidden_dim)

        # Bias for all entities
        self.b = nn.Parameter(torch.zeros(input_dim))

        print(f"Initialized ConvE components: conv_height={conv_height}, conv_width={conv_width}, flat_size={self.flat_size}")

    def calculate_score(self, embeddings, triplets):
        """
        Calculate the score for triplets using ConvE.
        ConvE scoring: f(h,r,t) = σ(vec(σ([M_h; M_r] * ω)) W) · t
        """
        batch_size = triplets.size(0)

        # Get embeddings
        h_emb = embeddings[triplets[:, 0]]  # [batch_size, hidden_dim]
        r_emb = self.relation_weights[triplets[:, 1]]  # [batch_size, hidden_dim]
        t_emb = embeddings[triplets[:, 2]]  # [batch_size, hidden_dim]

        # Reshape to 2D
        h_2d = h_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)
        r_2d = r_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)

        # Stack head and relation along height dimension
        stacked = torch.cat([h_2d, r_2d], dim=2)  # [batch_size, 1, 2*height, width]

        # Apply batch normalization and dropout
        x = self.bn0(stacked)
        x = self.input_dropout(x)

        # Apply 2D convolution
        x = self.conv1(x)  # [batch_size, out_channels, conv_height, conv_width]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten
        x = x.view(batch_size, -1)  # [batch_size, flat_size]

        # Project to embedding dimension
        x = self.fc(x)  # [batch_size, hidden_dim]
        x = self.output_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Score against tail entities (dot product)
        score = torch.sum(x * t_emb, dim=1) + self.b[triplets[:, 2]]  # [batch_size]

        return score

    def forward(self, graph, node_ids, rel_ids, norm):
        return self.rgcn(graph, node_ids, rel_ids, norm)

    def set_eval_mode_for_inference(self):
        """
        Set batch normalization layers to evaluation mode.
        CRITICAL for correct inference - ensures BN uses running statistics, not batch statistics.

        Call this before evaluation/inference, even if model.eval() was already called.
        """
        self.bn0.eval()
        self.bn1.eval()
        self.bn2.eval()

    def set_train_mode(self):
        """
        Set batch normalization layers to training mode.
        Call this when resuming training.
        """
        self.bn0.train()
        self.bn1.train()
        self.bn2.train()

    def regularization_loss(self, embeddings):
        """
        Compute regularization loss for embeddings and ConvE parameters.
        """
        return (torch.mean(embeddings.pow(2)) +
                torch.mean(self.relation_weights.pow(2)) +
                torch.mean(self.fc.weight.pow(2)) +
                torch.mean(self.b.pow(2)))

    def get_loss(self, graph, embeddings, triplets, labels):
        """
        Compute loss for link prediction, including regularization loss.
        """
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)
        return prediction_loss + self.regularization_param * reg_loss
