import soket

x = soket.Tensor(2, requires_grad=True)
y = soket.Tensor(5, requires_grad=True)

z = x.log() + x * y - y ** 2
print(y)
z.backward()

print(x.grad)
print(y.grad)

