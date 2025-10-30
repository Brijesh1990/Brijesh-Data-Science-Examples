#string list examples
list = ['apple', 'banana', 'mango']
x = input("Enter value to serch: ")

stringMatched = False
for i in range(len(list)):
    if list[i] == x:
       stringMatched = True
       break 

if stringMatched:
    print("Value match...")
else:
    print("Value not match...")