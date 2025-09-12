#prime numbers 

num = int(input("Enter a number to check if it is prime or not: "))
if num < 2:
    print(num, " is not prime number...")
else:
    a = True
    for i in range(2, num - 1):
        if num % i == 0:
            a = False
            break

    if a :
        print(num, " is prime number...")
    else:
        print(num, " is not prime number...")
        