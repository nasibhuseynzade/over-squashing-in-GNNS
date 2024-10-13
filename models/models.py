# Define the models for QM9
class RGATQM9(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_relations, num_tasks):
        super(RGATQM9, self).__init__()
        self.conv1 = RGATConv(num_node_features, hidden_channels, num_relations)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = nn.Linear(hidden_channels, num_tasks)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Convert edge_type to long tensor and get the first column if it's 2D
        if edge_type.dim() > 1:
            edge_type = edge_type[:, 0]
        edge_type = edge_type.long()

        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.relu(self.conv3(x, edge_index, edge_type))

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x

class RGINQM9(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_relations, num_tasks):
        super(RGINQM9, self).__init__()
        self.conv1 = RGINConv(num_node_features, hidden_channels, num_relations)
        self.conv2 = RGINConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGINConv(hidden_channels, hidden_channels, num_relations)
        self.lin = nn.Linear(hidden_channels, num_tasks)

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Convert edge_type to long tensor and get the first column if it's 2D
        if edge_type.dim() > 1:
            edge_type = edge_type[:, 0]
        edge_type = edge_type.long()

        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.relu(self.conv3(x, edge_index, edge_type))

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x