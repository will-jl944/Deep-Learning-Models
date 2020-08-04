import os
import csv
from config import config
import random


def main():
    with open('splited_data.csv', 'w') as sheet:
        writer = csv.writer(sheet)
        writer.writerow(['name', 'number'])
        for root, dirs, files in os.walk(config.data_folders[0]):
            for name in files:
                if '.png' in name and 'Img_Crop' in root and 'None' not in name:
                    if os.path.exists(os.path.join(config.pec_label_folder, name)):
                        split = random.uniform(0, 1)
                        writer.writerow([name, split])
    sheet.close()


if __name__ == '__main__':
    main()
