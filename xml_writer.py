from pathlib import Path
from lxml import etree as ET
import pandas as pd
import ps



def generate_labelimg_xml(annotations, image_path, width=1280, height=720, channel=3):
    image_path = Path(image_path)
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(channel)

    ET.SubElement(annotation, 'segmented').text = '0'

    id2ClassDict = {
        0: "mb2",
        1: "mb3",
        2: "mb4",
    }

    for i, row in annotations.iterrows():
        obj = ET.SubElement(annotation, 'object')
        obj_id, xmin, ymin, xmax, ymax  = row.values
        ET.SubElement(obj, 'name').text = id2ClassDict[obj_id]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)


    tree = ET.ElementTree(annotation)
    xml_file_name = image_path.parent / (image_path.name.split('.')[0]+'.xml')
    tree.write(str(xml_file_name), pretty_print=True)
