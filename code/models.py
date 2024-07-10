import torch
from torch import nn
from torch_geometric.nn import SAGEConv, GATConv, GINConv, GCNConv


class BipartiteSAGE2mod(torch.nn.Module):
    def __init__(self, nfeat:int, nproduct:int , hidden_channels:int , out_channels: int, num_layers:int, arch: str, dropout_rate:float =0):
        super().__init__()
        self.convs = torch.nn.ModuleList()

        self.user_embed = nn.Linear(nfeat, hidden_channels )
        self.item_embed =  nn.Linear(nproduct, hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.arch = arch
        print("Using ", self.arch, "...")
        for _ in range(num_layers):
            if self.arch == "sageconv":
                self.convs.append(SAGEConv((-1,-1), hidden_channels))
            elif self.arch == "gatconv":
                self.convs.append(GATConv((-1,-1), hidden_channels))
            elif self.arch =="gcnconv":
                self.convs.append(GCNConv(-1, hidden_channels))
            
        
        self.num_layers = num_layers

        self.hidden_common1 = nn.Linear(hidden_channels + num_layers*hidden_channels, hidden_channels)
        self.hidden_common2 = nn.Linear(hidden_channels, hidden_channels)

        self.hidden_control = nn.Linear(hidden_channels, int(hidden_channels/2))
        self.hidden_treatment = nn.Linear(hidden_channels, int(hidden_channels/2))

        self.out_control = nn.Linear( int(hidden_channels/2), out_channels)
        self.out_treatment = nn.Linear( int(hidden_channels/2), out_channels)

        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.activation = nn.ReLU()


    def forward(self, xu: torch.tensor, xp:torch.tensor, edge_index:torch._tensor):
        out = [] 
        xu = self.user_embed(xu)
        xp = self.item_embed(xp)

        out.append(xu)

        embeddings = torch.cat((xu,xp), dim=0) 
        
        for i in range(self.num_layers):
            embeddings = self.activation(self.convs[i](embeddings, edge_index))
            
            out.append(embeddings[:xu.shape[0]])            
        
        out = torch.cat( out, dim=1)
        
        hidden = self.dropout(self.activation(self.hidden_common1(out)))
        hidden = self.dropout(self.activation(self.hidden_common2(hidden)))
        
        hidden_1t0 = self.dropout(self.activation(self.hidden_control(hidden)))
        hidden_1t1 = self.dropout(self.activation(self.hidden_treatment(hidden)))

        out_2t0 = self.activation(self.out_control(hidden_1t0))
        out_2t1 = self.activation(self.out_treatment(hidden_1t1))
        
        return out_2t1, out_2t0, hidden_1t1, hidden_1t0