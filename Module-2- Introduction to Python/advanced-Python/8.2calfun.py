def cal():
    number1=int(input("Enter first number "))
    number2=int(input("Enter second number "))

    result=0

    # Calculator logic
    if method == 1:
        result=addition(number1,number2)
    elif method == 2:
        result=substraction(number1,number2)
    elif method == 3:
        result=multiplication(number1,number2)
    elif method == 4:
        result=divison(number1,number2)
    else:
        result=modulo(number1,number2)

    print("result value: ", result)


def addition(a:int,b:int) -> int:   # addition function
    return a+b

def substraction(a:int,b:int) -> int:   # substraction function
    return a-b

def multiplication(a:int,b:int) -> int: # multiplication function
    return a*b

def divison(a:int,b:int) -> int:    # divison function
    return a/b

def modulo(a:int,b:int) -> int: # modulo function
    return a%b


while True:
    print("Please choose any method...\n\n1 for Addition \n2 for Substraction \n3 for Multiplication \n4 for Divison \n5 for Modulo \n6 for Exit")

    # User input
    method=int(input("Enter any method... "))
    if method in range(1,7):
        if method == 6:
            #   quit()
            break
        else:
            cal()
            continue
    else:
        print("Please Choose valid method\n")
        continue

