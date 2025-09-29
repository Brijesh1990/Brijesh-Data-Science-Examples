age = int(input("Enter your age: "))
weight = int(input("Enter your weight: "))

if age in range(18,65):
    if weight >= 45:
        print("You are eligible  for blood donate")
    else:
        print("You are not eligible  for blood donate") 
else:
    print("You are not eligible  for blood donate")