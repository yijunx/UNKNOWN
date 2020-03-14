import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

print(z.requires_grad)

a = torch.randn(2, 2)
print(a)
b = ((a * 3) / (a - 1))
print(b)
a = ((a * 3) / (a - 1))
print(a)

print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.requires_grad)
print(b.grad_fn)



# gradients
out.backward()
print(out.grad)
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)
