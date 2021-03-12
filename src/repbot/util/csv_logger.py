# Author: Javad Amirian
# Email: amiryan.j@gmail.com

from datetime import datetime


class CsvLogger:
    def __init__(self, filename, filemode='a'):
        self.filename = filename
        self.filemode = filemode

    def log(self, array, add_timestamp=True):
        with open(self.filename, mode=self.filemode) as csv_file:
            if add_timestamp:
                now = datetime.now().strftime("%Y-%b-%d|%H:%M:%S")
                csv_file.write(now+', ')
            for i, arr in enumerate(array):
                if i != len(array) - 1:
                    csv_file.write(str(arr) + ", ")
                else:
                    csv_file.write(str(arr) + "\n")


