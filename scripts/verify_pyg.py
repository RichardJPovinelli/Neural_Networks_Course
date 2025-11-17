import sys

print('Python:', sys.executable)
try:
    import torch
    print('torch version:', torch.__version__)
    print('torch built cuda:', getattr(torch.version, 'cuda', None))
    print('torch.cuda.is_available():', torch.cuda.is_available())
except Exception as e:
    print('Error importing torch:', e)

try:
    import torch_geometric
    print('torch_geometric version:', getattr(torch_geometric, '__version__', None))
    from torch_geometric.datasets import Planetoid
    print('Planetoid import: OK')
except Exception as e:
    print('Error importing torch_geometric or Planetoid:', e)

print('\nDone.')
