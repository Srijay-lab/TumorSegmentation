"""
Created on Tue Nov 26 16:45:17 2019
This simple script allows you to visualize a prediction mask as polygon annotations in ASAP
Possible Usecase:
    You have performed patch level prediction and used these predictions to construct a binary mask.
    You want to overlay the mask as an annotation on the whole slide image in ASAP.
Dependencies:
    For running the code:
        pip install imantics
        numpy
        xml
    For visualization: ASAP (https://computationalpathologygroup.github.io/ASAP/)
    If you want the code to automatically figure out WSI dimensions, specify ASAP_PATH
    Tested with both Linux (10.04 Ubuntu) and Windows 10 in Python 3.7
@author: fayyaz (Dr. Fayyaz Minhas https://sites.google.com/view/fayyaz/home)
"""

import numpy as np
from imantics import Mask
import xml.etree.cElementTree as ET
import random
from openslide import open_slide
import scipy.misc

def writeASAPPolygon(polygons, ofname="annot.xml", color="#F4FA58"):
    if type(color) == type(""):
        color = [color] * len(polygons)
    root = ET.Element("ASAP_Annotations")
    doc = ET.SubElement(root, "Annotations")
    for i, px in enumerate(polygons):
        p = ET.SubElement(doc, "Annotation", name="Annotation " + str(i), Type="Polygon", PartOfGroup="None",
                          Color=color[i])
        c = ET.SubElement(p, "Coordinates")
        rpx = px.reshape((-1, 2))
        for j, (x, y) in enumerate(rpx):
            ET.SubElement(c, "Coordinate", Order=str(j), X=str(x), Y=str(y))
    tree = ET.ElementTree(root)
    tree.write(ofname)

def mask2ASAPAnnotation(mask, ofname="annot.xml", color=[], wsi_path=None, wsi_wh=(80e3, 80e3), offset=(0, 0), stol=0):
    """
    Given a prediction mask, generate an ASAP polygon annotation xml file
    Inputs:
        mask: numpy array mask with integer values (e.g., 0 for backgnd, 1 for type 1, 2 for type 2 etc)
        ofname: output file name
        color: color of annotation. (either [] or must contain hex colors equal to the number of types in the mask)
        wsi_path: path of the corresponding WSI file - if given, the method tries to use ASAP multiresolutionimageinterface to automatically get WSI image size
        wsi_wh: (width,height) of corresponding WSI. Either wsi_wh or wsi_path must be given in order to calculate appropriate rescaling factors
        offset: offset (x,y) of the mask from the top left (0,0) in the WSI
        stol: polygon simplification tolerance (default 100)
    Outputs:
        Writes an XML annotation file that can be loaded into ASAP for visualization
        Reurns the polygons
    """
    offset_x, offset_y = offset
    mask_y, mask_x = mask.shape

    if wsi_path is not None:
        try:
            slide = open_slide(wsi_path)
            wsi_wh = slide.dimensions
            print(mask.shape)
            print(wsi_wh)
        except Exception as e:
            print(e)
            print("Warning: Unable to get WSI dimensions. No ASAP?\nUsing dimensions:", wsi_wh)

    wsi_x, wsi_y = wsi_wh
    rescale_x = wsi_x / mask_x  # based on WSI_x/s_x
    rescale_y = wsi_y / mask_y  # based on WSI_y/s_y

    print("Rescaling x factor => ",rescale_x)
    print("Rescaling y factor => ",rescale_y)

    U = list(np.unique(mask))
    U.remove(0.0)
    polygons = []
    number_of_colors = len(U)
    if not color:
        C = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    else:
        C = color
    assert (number_of_colors == len(C))
    color = []
    for i, u in enumerate(U):
        poly_u = Mask(mask == u).polygons()
        polygons.extend(poly_u)
        color.extend([C[i]] * len(list(poly_u)))

    # rescale polygon coordinates
    polygons = [p.reshape((-1, 2)) for p in polygons]
    wsi_polygons = []  # polygons for the whole slide image
    for p in polygons:
        p[:, 0] = p[:, 0] * rescale_x + offset_x
        p[:, 1] = p[:, 1] * rescale_y + offset_y
        wsi_polygons.append(p)
    if stol == 0:
        retpols = wsi_polygons
    else:
        from shapely import geometry
        retpols = []
        for p in wsi_polygons:
            try:
                p = np.array(geometry.Polygon(p).simplify(stol, preserve_topology=True).exterior.coords)
            except:
                pass
            retpols.append(p)
    writeASAPPolygon(retpols, ofname, color)
    return retpols

def GenerateASAPXML(wsi_path, xml_file_name):
    mask = scipy.misc.imread("binary_mask.png")
    mask2ASAPAnnotation(mask, wsi_path=wsi_path, ofname=xml_file_name, stol=0)