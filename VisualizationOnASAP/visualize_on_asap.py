from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import math
import numpy as np
import scipy.misc
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
from tensorflow.keras.preprocessing import image
from VisualizationOnASAP.generate_ASAP_xml import GenerateASAPXML
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input
import argparse

def LoadModel(trained_model_path):
    img_shape = (448, 448, 3)
    # load pre-trained resnet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=img_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Flatten(name='flatten')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.50)(x)
    x = Dense(2, activation='softmax', name='fc8')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(trained_model_path)
    for layer in model.layers:
        layer.trainable = False
    print("model is loaded")
    return model

def GenerateMask(wsi_path,trained_model_path,patch_size,magnification,min_color_threshold,max_color_threshold):
    slide = open_slide(wsi_path)
    tiles = DeepZoomGenerator(slide,
                              tile_size=patch_size,
                              overlap=0,
                              limit_bounds=True)
    level = tiles.level_count - int(math.log((40/magnification),2)) - 1
    x_tiles, y_tiles = tiles.level_tiles[level]
    model = LoadModel(trained_model_path)
    binary_mask = np.zeros((y_tiles,x_tiles))
    x, y = 0, 0
    while y < y_tiles:
        while x < x_tiles:
            new_tile = np.array(tiles.get_tile(level, (x, y)), dtype=np.uint8)
            avg = np.average(new_tile)
            if (min_color_threshold <= avg <= max_color_threshold):
                if np.shape(new_tile) == (patch_size, patch_size, 3):
                    if not os.path.exists('Intermediate_Tiles'):
                        os.makedirs('Intermediate_Tiles')
                    filename = "./Intermediate_Tiles/"+str(x)+"_"+str(y)+".png"
                    scipy.misc.imsave(filename, new_tile)
                    test_image = image.load_img(filename, target_size=(448, 448))
                    test_image = image.img_to_array(test_image)
                    test_image = np.expand_dims(test_image, axis=0)
                    test_image = preprocess_input(test_image)
                    prob = model.predict(test_image,steps=1)
                    pred=prob.argmax(axis=1)
                    pred=pred[0]
                    if(pred == 0): #If invasive class
                        binary_mask[y][x]=255
            x+=1
        y+=1
        x=0

    scipy.misc.imsave('binary_mask.png', binary_mask)

def action(args):
    wsi_path = args.wsi_path
    patch_size = args.patch_size
    magnification = args.magnification
    trained_model_path = args.trained_model_path
    min_color_threshold = args.min_color_threshold
    max_color_threshold = args.max_color_threshold

    xml_file_name = Path(wsi_path).stem + ".xml"

    GenerateMask(wsi_path=wsi_path,
                 trained_model_path=trained_model_path,
                 patch_size=patch_size,
                 magnification=magnification,
                 min_color_threshold=min_color_threshold,
                 max_color_threshold=max_color_threshold)

    GenerateASAPXML(wsi_path=wsi_path, xml_file_name=xml_file_name)

    return 0

print("Let us visualize predictions on ASAP")
parser = argparse.ArgumentParser(description='Deep Neural Networks')
parser.add_argument('--wsi-path', default="F:/Datasets/HER2C/Data/Train/0/01_HE.ndpi", help='path to WSI file')
parser.add_argument('--trained-model-path', default="C:/Users/Srijay/Desktop/Projects/Keras/SegmentTumors/Models/herohe/final.ckpt", help='path to trained model')
parser.add_argument('--patch-size', default=512, help='Patch Size')
parser.add_argument('--magnification', default=20, help='Magnification')
parser.add_argument('--min-color-threshold', default=100, help='min_color_threshold')
parser.add_argument('--max-color-threshold', default=205, help='max_color_threshold')
action(parser.parse_args())


