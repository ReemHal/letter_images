
# For each file in the given dataset, identify the different parts of the letter and save it in the contours directory')

import os
import argparse
import letter_image as li

parser = argparse.ArgumentParser(description='For each letter image in dataset_dir, identify the different parts of the letter and save it in the contours directory')
parser.add_argument('-d', '--dataset', required=False,
                    help='dataset directory containing the letter images',
                    default= "/media/datadr/datasets/letters")
parser.add_argument('-s', '--savedir', required=False,
                    help='directory where the letter parts are saved',
                    default= "contours")

args = parser.parse_args()

dataset_dir = args.dataset
savedir_name = args.savedir

if not(os.path.isdir(savedir_name)):
    os.makedirs(savedir_name)

for file in os.listdir(dataset_dir):
    print("processing",file)
    letter_obj = li.letter_image(dataset_dir, file)
    letter_obj.process_letter()
    letter_obj.display_contours(display=False,save=True,savedir='contours')
