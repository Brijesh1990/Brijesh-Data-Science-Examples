#loop
even_num_generator = (i for i in range(1, 21) if i % 2 == 0)
for i in even_num_generator:
    print(i)