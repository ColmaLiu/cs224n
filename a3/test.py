import torch

embeddings = torch.arange(20).reshape(5,4)
w = torch.tensor([[0,3],
     [2,4],
     [1,4]])
w_expanded = w.unsqueeze(-1).expand(-1, -1, 4)
x = torch.flatten(torch.gather(embeddings.unsqueeze(0).expand(3,-1,-1), 1, w_expanded), 1)
print(embeddings)
print(x)