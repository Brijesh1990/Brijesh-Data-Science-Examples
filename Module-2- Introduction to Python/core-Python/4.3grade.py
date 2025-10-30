maths = float(input("Enter maths marks : "))
physics = float(input("Enter physics marks : "))
chemistry = float(input("Enter chemistry marks : "))

total = maths + physics + chemistry
per = total / 3
print("You have percentage ", per)
if per >= 90:
    print("You have got A+ grade...")
elif per >= 80: 
    print("You have got A grade...")
elif per >= 70: 
    print("You have got B+ grade...")
elif per >= 60: 
    print("You have got B grade...")
elif per >= 50: 
    print("You have got C+ grade...")
elif per >= 40: 
    print("You have got C grade...")
else: 
    print("Fail...")   