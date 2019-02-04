"""
for every letter image in the dataset directory:
        1. extract the contents of the letter
        2. identify the different parts including: header, recipient, date, subject, greeting, body, signature, and enclusures/notes.
        3. store identified information in a csv file.
"""

import os
import argparse
import csv
import copy
import letter_image as li

def get_letter_info(letter_obj, letter_info):
    """
        get the current letter's data such as the letter parts, text in each part,
        and the contours marking this info in the  image.

        letter_obj: an instantiated letter_image object
        letter_info: a dictionary containing the dir and image_name of the current letter image
    """

    letter_info_list = []
    keys=[]
    parts=[]
    for i, clip_item in enumerate(letter_obj.clip):
        curr_info_dict = copy.deepcopy(letter_info)
        curr_info_dict.update(clip_item)
        parts = parts+ list({clip_item['part']})
        keys= keys + list(clip_item.keys())
        letter_info_list.append(curr_info_dict)
    return letter_info_list, keys

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='For each letter image in dataset_dir, idetify the different parts of the letter and save it in the contours directory')
    parser.add_argument('-d', '--dataset', required=False,
                        help='dataset directory containing the letter images',
                        default= "/media/datadr/datasets/letters")
    parser.add_argument('-s', '--savedir', required=False,
                        help='directory where the letter parts are saved',
                        default= "contours")
    parser.add_argument('-i', '--info', required=False,
                        help='file name where the resulting data is saved',
                        default= "infoTable.csv")

    args = parser.parse_args()

    dataset_dir = args.dataset
    savedir_name = args.savedir
    info_file_name = args.info

    if not(os.path.isdir(savedir_name)):
        os.makedirs(savedir_name)

    letter_info_list = []
    keys=[]
    for file in os.listdir(dataset_dir):
        print("processing",file)
        letter_obj = li.letter_image(dataset_dir, file)
        if not(letter_obj is None):
            letter_obj.process_letter()
            letter_info_dict = {'image_name': file, 'dir': dataset_dir}
            curr_list, curr_keys = get_letter_info(letter_obj, letter_info_dict)
            letter_info_list= letter_info_list+curr_list
            keys = keys+curr_keys
            letter_obj.display_contours(display=False, save=True, savedir='contours')

    keys = ['image_name', 'dir'] + list(set(keys))

    # make sure all rows have all keys
    for i, row in enumerate(letter_info_list):
        missing_keys = set(keys) - set(row.keys())
        for key in missing_keys:
            row.update({key: None})
        letter_info_list[i] = row

    # save letter_info_list in csv file
    with open(info_file_name, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(letter_info_list)
