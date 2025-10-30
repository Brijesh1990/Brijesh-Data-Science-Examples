# map using functions 
def square(n):
    return n * n

a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
b = map(square, a)

print(list(b))