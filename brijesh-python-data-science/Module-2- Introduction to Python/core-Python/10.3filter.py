def even(n):
    return n % 2 == 0

a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
b = filter(even, a)

print(list(b))