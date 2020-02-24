import csv


# A class for creating a file writer, writing rows in a csv-file
from datetime import datetime


class FileWriter:

    def __init__(self, path, col_names):
        # Set filename to timestamp and set path
        timestamp = datetime.now().strftime("%d:%m:%Y:%H-%M-%S")
        file_path = path + timestamp + '.csv'

        # Create writer
        self.file = open(file_path, mode='w')
        self.writer = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write column names as first row in file
        self.writer.writerow(col_names)

    # Write a list of data as a row in the file
    def write_to_file(self, data):
        self.writer.writerow(data)
