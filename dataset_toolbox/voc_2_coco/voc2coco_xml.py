import json
import os
from PIL import Image
import xml.etree.ElementTree as ET
import shutil

def parse_xml(xml_path):
    tree=ET.parse(xml_path)
    root=tree.getroot()
    objs=root.findall('object')
    coords=list()
    for ix,obj in enumerate(objs):
        name=obj.find('name').text
        box=obj.find('bndbox')
        x_min=int(box[0].text)
        y_min=int(box[1].text)
        x_max=int(box[2].text)
        y_max=int(box[3].text)
        coords.append([x_min,y_min,x_max,y_max,name])
    return coords

def convert(root_path,source_xml_root_path,target_json_root_path,phase='train',split=8000):
    dataset={'categories':[],'images':[],'annotations':[]}

    classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair',
             'cow','diningtable','dog','horse','motorbike','person','pottedplant',
             'sheep','sofa','train','tvmonitor']

    for i,cls in enumerate(classes):
        dataset['categories'].append({'id':i,'name':cls,'supercategory':'beverage'})

    images=[f for f in os.listdir(os.path.join(root_path,'JPEGImages/'))]
    if phase=='train':
        images=[line for i,line in enumerate(images) if i<=split]
    elif phase=='test':
        images=[line for i,line in enumerate(images) if i>split]

    print('----------------------------------start_convert--------------------------------------------')

    names=[]
    bnd_id=1

    for i,image in enumerate(images):
        xml_path=os.path.join(source_xml_root_path,image[:-4]+'.xml')
        image_path=os.path.join(root_path,'JPEGImages/'+image)

        try:
            img=Image.open(image_path)
            height=img.height
            width=img.width
        except(OSError,NameError):
            print('OSError,Path:',image_path)
            os.remove(image_path)
            continue
        dataset['images'].append({'file_name':image,'id':i,'width':width,'height':height})
        try:
            coords=parse_xml(xml_path)
        except:
            print(image[:-4]+'.xml not exists~')
            continue
        for coord in coords:
            x1=int(coords[0])-1
            x1=max(x1,0)

            y1=int(coords[1])-1
            y1=max(y1,0)

            x2=int(coords[2])
            y2=int(coords[3])

            name=coord[4]
            names.append(name)
            cls_id=classes.index(name)+1
            width=max(0,x2-x1)
            height=max(0,y2-y1)
            dataset['annotations'].append({
                'area':width*height,
                'bbox':[x1,y1,width,height],
                'category_id':int(cls_id),
                'id':bnd_id,
                'image_id':i,
                'iscrowd':0,
                'segmentation':[[x1,y1,x2,y1,x2,y2,x1,y2]]
            })
            bnd_id+=1
    folder=os.path.join(target_json_root_path,'annotations_json')
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    json_path=os.path.join(target_json_root_path,'annotations_json/instances_{}2017.json',format(phase))

    with open(json_path,'a') as f:
        json.dump(dataset,f)

if __name__=='__main__':
    convert(root_path='/home/root',source_xml_root_path='/home/xml_root',target_json_root_path='/home/json_root')