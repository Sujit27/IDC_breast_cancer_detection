import os
import glob
'''Creates a txt file with all the image file names present in the datapath'''

data_path = "dataset/IDC_regular_ps50_idx5"
output_file = "annotations.txt"


all_patients = os.listdir(data_path)
list_of_images = []

for patient in all_patients:
    list_of_negative_images = glob.glob(os.path.join(data_path,patient,"0","*.png"))
    list_of_positive_images = glob.glob(os.path.join(data_path,patient,"1","*.png"))
    list_of_images.append(list_of_negative_images)
    list_of_images.append(list_of_positive_images)

list_of_images = [image_file for section in list_of_images for image_file in section]

with open(output_file,'w') as f:
    for image_file in list_of_images:
        f.write("%s\n" %image_file)
