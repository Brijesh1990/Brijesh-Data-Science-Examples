list= ['apple', 'banana', 'mango']
i=0
while i < len(list):
    if list[i] == "banana":
        i+=1
        continue
    else:
        print(list[i])
        i+=1

