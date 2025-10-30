list1=["a", "b", "c"]
list2=[101, 102, 103]
dictionary={}
for i in range(0, len(list1)):
    dictionary[list1[i]] = list2[i]
print(dictionary)