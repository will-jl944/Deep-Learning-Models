import csv
import random


def get_info_dict(csv_path, random_win_lvl):
    info_dict = {}
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0].strip():
                fake_id = row[1]
                instance_num = row[2]
                view_pos = row[3]
                img_lat = row[4]

                window_center = row[14]
                window_width = row[15]
                if len(window_width) > 4 and len(window_center) > 4:
                    window_width_list = [float(w.strip("'")) for w in window_width.strip('[]').split(', ')]
                    max_width = max(window_width_list)
                    i = window_width_list.index(max_width)
                    window_center_list = [float(c.strip("'")) for c in window_center.strip('[]').split(', ')]
                    max_center = window_center_list[i]
                elif window_width and window_center and len(window_width) < 4 and len(window_center) < 4:
                    max_width = float(window_width)
                    max_center = float(window_center)
                else:
                    pass

                info_dict[fake_id+'_'+img_lat+'_'+view_pos+'_'+instance_num] = (max_center, max_width)

    return info_dict


def get_coord_dict(csv_path):
    coord_dict = {}
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if 'DN' in row[0]:
                coord = (float(row[1].strip('()').split(', ')[0]), float(row[1].strip('()').split(', ')[1]))
                coord_dict[row[0]] = coord

    return coord_dict


def get_train_test_dict(csv_path):
    train_test_dict = {}
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0] == 'name':
                continue
            else:
                train_test_dict[row[0]] = float(row[1])
    return train_test_dict
