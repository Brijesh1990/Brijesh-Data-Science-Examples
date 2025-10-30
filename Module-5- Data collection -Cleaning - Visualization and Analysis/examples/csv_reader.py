import csv

# opening the CSV file
with open('Companies.csv', newline='') as file:
    # using the reader() function to read the content of the file
    reader = csv.reader(file)

    # printing each row of the table
    for row in reader:
        print(row)