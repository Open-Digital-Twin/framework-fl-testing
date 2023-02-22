from os import listdir
from os import path as Path

# Function to Check if the path specified
# specified is a valid directory
def isEmpty(path):
    if Path.exists(path) and not Path.isfile(path):
  
        # Checking if the directory is empty or not
        if not listdir(path):
            return True
        else:
            return False
    else:
        return True
  