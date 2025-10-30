import csv    
# List of dictionaries representing rows    
data = [    
    {'Name': 'John', 'Age': 20, 'City': 'Hyderabad'},    
    {'Name': 'Sachin', 'Age': 21, 'City': 'Pune'},    
    {'Name': 'Lucy', 'Age': 40, 'City': 'New York'}    
]    
    
# Writing to CSV using DictWriter    
with open('dictwriter-example.csv', 'w', newline='') as file:    
    fieldnames = ['Name', 'Age', 'City']    
    writer = csv.DictWriter(file, fieldnames=fieldnames)    
        
    writer.writeheader()        # Write header row    
    writer.writerows(data)      # Write all rows    