from functools import reduce

def product(x, y):
    return x * y

a = [1, 2, 3, 4]
b = reduce(product, a)

print(b)