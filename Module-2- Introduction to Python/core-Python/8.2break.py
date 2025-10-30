list= ['apple', 'banana', 'mango']
i=0
while i < len(list):
    if list[i] == "banana":
        i+=1
        break
    else:
        print(list[i])
        i+=1

