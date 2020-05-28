import os
import shutil

source_image_root_path='./data_voc/VOC2007/JPEGImages'
target_image_root_path='./data_coco/train2017'

if os.path.exists(target_image_root_path):
    shutil.rmtree(target_image_root_path)
os.makedirs(target_image_root_path)

for parent,_,files in os.walk(source_image_root_path):
    for file in files:
        source_image_path=os.path.join(source_image_root_path,file)
        target_image_path=os.path.join(target_image_root_path,file)
        shutil.copyfile(source_image_path,target_image_path)
print('finished!')