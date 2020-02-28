import csv


# A class for creating a file writer, writing rows in a csv-file
from datetime import datetime


class FileWriter:

    def __init__(self, path, title):
        # Set filename to timestamp and set path
        timestamp = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
        file_path = path + timestamp + '.csv'

        # Create writer
        self.file = open(file_path, mode='w')
        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write column names as first row in file
        self.writer.writerow([title, timestamp])

    # Write a list of data as a row in the file
    def write_to_file(self, data):
        self.writer.writerow(data)

    # Close the file
    def close(self):
        self.file.close()
