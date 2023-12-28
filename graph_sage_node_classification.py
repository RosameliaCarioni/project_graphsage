import torch
from torch_geometric.datasets import Reddit
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
import time
from sklearn.metrics import f1_score 
import graph_handler

def accuracy(pred_y, y): 
    return ((pred_y == y).sum()/len(y)).item()

class GraphSAGE_local(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, aggr='mean', normalization = True, activation_function = True, bias = True):

        super().__init__()
        # as K = 2, we have 2 layers
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, out_channels = hidden_channels, project = activation_function, bias = bias)
        self.conv2 = SAGEConv(hidden_channels, out_channels = out_channels, project = activation_function, bias = bias, normalization = normalization)
    

    def forward(self, matrix_nodes_features, edge_index):
      # matrix_nodes_features is a matrix from the data where row = nodes, columns = feature
      # edge_index: This is a tensor that describes the connectivity of the graph. Each column in this matrix represents an edge. The first row contains the indices of the source nodes, and the second row contains the indices of the target nodes.
    
        h = self.conv1(matrix_nodes_features, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training = self.training)

        h = self.conv2(h, edge_index)
        h = F.relu(h) # TODO: maybe remove this 
        h = F.dropout(h,  p=self.dropout, training = self.training) # TODO: maybe remove this
        h = F.log_softmax(h, dim = 1)
        return h
    
    def fit(self, loader, epochs, learning_rate, device):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.train()
        times = []

        for epoch in range(epochs+1):
            start = time.time()
            train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0 
            
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = self(batch.x, batch.edge_index) # obtain the probability of belonging to each class or label for each node 
                
                loss = criterion(out[batch.train_mask],  batch.y[batch.train_mask]) 
                
                # Train data
                train_loss += loss.item()
                train_acc += accuracy(out[batch.train_mask].argmax(dim = 1), batch.y[batch.train_mask])

                loss.backward()
                optimizer.step()

                # Validation data
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
                val_acc += accuracy((out[batch.val_mask]).argmax(dim = 1), batch.y[batch.val_mask]) 

            # All following values are average per batch 
            print(f'Epoch {epoch:>3} | Train Loss: {train_loss/len(loader):.3f} | Train Acc: {train_acc/len(loader)*100:>6.2f}% | Val Loss: {val_loss/len(loader):.2f} | Val Acc: {val_acc/len(loader)*100:.2f}%')
          
            times.append(time.time() - start)
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
        
    @torch.no_grad()
    def test(self, data):
        self.eval()
        out = self(data.x, data.edge_index)
        y = data.y[data.test_mask]
        y_prediction = out.argmax(dim = 1)[data.test_mask]

        acc = accuracy(y_prediction, y)
        f1_macro = f1_score(y, y_prediction, average = 'macro')
        f1_micro =  f1_score(y, y_prediction, average = 'micro')
        return acc, f1_macro, f1_micro