# from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)  # empty will shown values were in the allocated memory at the time

x = torch.rand(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.tensor([5.5, 3])
print(x)

x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.mean(), x.size())

y = torch.rand(5, 3)
print(y)
print(x + y)

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)  # add in place
print(y)
# ny operation that mutates a tensor in-place is post-fixed with an _

print(x)
print(x[:, -1])
print(x[-1, :])


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions -1 means 4*4/8 => 2
print(x.size(), y.size(), z.size())

x = torch.randn(1)
print(x)
print(x.item())

if torch.cuda.is_available():
    print('yo')
