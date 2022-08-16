import sys, os
# sys.path.insert(0, '/content/drive/MyDrive/CloudsDL/BigEarthNet_exe_data/bigearthnet-models-tf-master')
# from BigEarthNet import BigEarthNet

import numpy as np
import pandas as pd
import re

# import tensorflow.compat.v1 as tf
import subprocess, time, os
import argparse
import json
import glob
import importlib
from osgeo import gdal, osr
gdal.UseExceptions()
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import geopandas as gpd
import shapely
from pprint import pprint
from PIL import Image

import cv2
from tqdm import tqdm
from skimage.util.shape import view_as_windows

import tensorflow as tf
from tensorflow import keras
import pandas as pd

tiles = pd.read_csv('E:\\Users\\sentinel_industry\\deploy/but_nanjing', index_col=0)
scene_ids = tiles.index.to_numpy()

output_layers = 'deploy/deploy_outl/'
image_download_folder = 'deploy/deploy_dwnimg/'
out_put_blocks = 'deploy/deploy_blk'


os.environ['GS_NO_SIGN_REQUEST']='YES'
os.environ['AWS_NO_SIGN_REQUEST']='YES'
os.environ['AWS_VIRTUAL_HOSTING'] ='FALSE'
os.environ['AWS_HTTPS'] = 'YES'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = ".tif,.tiff,.jp2,.mrf,.idx,.lrc,.mrf.aux.xml,.vrt"
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'YES'



def get_asw_s2_path(scid):
    http_prefix = '/vsicurl/http://sentinel-cogs.s3.amazonaws.com/sentinel-s2-l2a-cogs/'
    
    http_path = http_prefix+\
                re.findall(r'\d+',scid.split('_')[1])[0]+'/'+\
                re.findall("[a-zA-Z]+", scid.split('_')[1])[0][0]+'/'+\
                re.findall("[a-zA-Z]+", scid.split('_')[1])[0][1:3]+'/'+\
                scid[10:14]+'/'+\
                str(int(scid[14:16]))+'/'+\
                scid+'/' 
    return http_path

def download_worker_bands(payload):
        
        payload['download_status'] = []
        try:
            if payload['ep'] == 'aws_cog':
                
                path=payload["aws_path"]+payload['download_band']+'.tif'
                now_ras = gdal.Open(path)
                gdal.GetDriverByName('GTiff')\
                    .CreateCopy(str(Path(payload['outfolder'], 
                                        payload['id']+"_"+payload['download_band']+'.tif')), now_ras)
                now_ras=None
                payload['download_status'] = payload['download_status']+['from aws to : '
                                        +str(Path(payload['outfolder'], 
                                        payload['id']+"_"+payload['download_band']+'.tif'))]
        except RuntimeError as err:
            print('Dang... ')

def download_worker(payload):
    task_list = []
    for band in payload['download_bands']:
        payload['download_band'] = band
        task_list.append(copy.deepcopy(payload))
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(download_worker_bands, task_list)


def download_df(scene_df, output_folder, bands_l2a = None, 
                                        end_point='aws_cog'):
        # Download for AWS
        if end_point == 'aws_cog':

            bands = [band for band in bands_l2a if bands_l2a[band]]
            #download_df['download_bands'] = bands
            scene_df['outfolder'] = output_folder
            scene_df['ep'] = end_point
            task_list = list(scene_df.T.to_dict().values())

            for tsk in task_list:
                tsk['download_bands'] =bands 
            with ThreadPoolExecutor(max_workers=4) as executor:
                tqdm(executor.map(download_worker, task_list), total=len(task_list))
            #scene_df = pd.DataFrame(task_list)

def output_image(output_img, nowGt, ImPrj, arr, data_type):

        nowBlockDs = gdal.GetDriverByName('GTiff').Create(output_img, 
                                                            arr.shape[2], 
                                                            arr.shape[1],
                                                            arr.shape[0],
                                                            data_type)
                                                            #gdal.GDT_UInt16)
        nowBlockDs.SetGeoTransform(nowGt)
        nowBlockDs.SetProjection(ImPrj)
        
        for i in range(arr.shape[0]):
            nowBlockBand = nowBlockDs.GetRasterBand(i+1)
            nowBlockBand.WriteArray(arr[i])
            nowBlockBand.SetNoDataValue(0)
            nowBlockBand.FlushCache()
        nowBlockDs=None

def bounds_to_polygon(bounds):
    '''
    function that converts a bounds list to a shapely Polygon object
    bounds :: list :: [xmin, ymin, xmax, ymax]
    return :: Shapely Polygon object
    '''
    return shapely.geometry.Polygon([[bounds[0],bounds[3]],
                    [bounds[0],bounds[1]],
                    [bounds[2],bounds[1]],
                    [bounds[2],bounds[3]]])

# Download images
df_list=[]
for scene in scene_ids:
    if not os.path.isfile(image_download_folder+scene+'_B02.tif'):
        df_list.append([scene, get_asw_s2_path(scene)])
    else: print('already done for: ', scene)
        
dwnld_df = pd.DataFrame(df_list, columns=['id','aws_path'])

platform_bands_l2a={"B02": True,"B03": True,
                    "B04": True,"B08":True}
output_folder=image_download_folder
download_df(dwnld_df, output_folder,  bands_l2a = platform_bands_l2a, end_point='aws_cog')

base_dir = image_download_folder

images_df_list = [[("_").join(x.split("_")[1:3]),
                    x.split("_")[-1].split('.')[0],
                    os.path.join(base_dir,x)] for x in os.listdir(base_dir) if x.endswith('.tif')]
images_df = pd.DataFrame(images_df_list, columns= ['id', 'band', 'path'])


####################################################################
# Optional
####################################################################
# Save RGB images for visuals
for scene in scene_ids:
    stbits = scene.split('_')
    shortname = stbits[1]+'_'+stbits[2]
    shortname
    if not os.path.isfile(os.path.join(output_layers,shortname+'_rgb.tif')):
        print('saving visuals for: ', scene)
      # red
        redDs = gdal.Open(os.path.join(image_download_folder,scene+'_B04.tif'))
        redArr = redDs.ReadAsArray()
        nowGt = redDs.GetGeoTransform()
        ImPrj = redDs.GetProjection()
        redDs=None
        rgb = redArr[np.newaxis,...]
        # green
        greenDs = gdal.Open(os.path.join(image_download_folder,scene+'_B03.tif'))
        greenArr = greenDs.ReadAsArray()
        greenDs=None
        rgb = np.append(rgb, greenArr[np.newaxis,...], axis=0)
        # blue
        blueDs = gdal.Open(os.path.join(image_download_folder,scene+'_B02.tif'))
        blueArr = blueDs.ReadAsArray()
        blueDs=None
        rgb = np.append(rgb, blueArr[np.newaxis,...], axis=0)
    #     if os.path.isdir(os.path.join(output_layers, scene)) == False:
    #         os.makedirs(os.path.join(output_layers, scene))
        
        output_image(os.path.join(output_layers,shortname+'_rgb.tif'), nowGt, ImPrj, rgb, gdal.GDT_UInt16)

    else: print('already done for: ', scene)
        
        
####################################################################
# BLOCKS
####################################################################


band_order = ["B02","B03","B04","B08",]
s2_dims = {10: 10980, 20: 5490, 60: 1830}

in_shape={'B01': 20,
          'B02': 120,
          'B03': 120,
          'B04': 120,
          'B05': 60,
          'B06': 60,
          'B07': 60,
          'B08': 120,
          'B09': 20,
          'B11': 60,
          'B12': 60,
          'B8A': 60,}

uni_scenes = images_df['id'].unique()
for uscn in uni_scenes:
    print('For: ', uscn)
    if os.path.isdir(os.path.join(out_put_blocks, uscn)) == True:
        print('already blocked for: ', uscn)
        continue
    else:
        os.makedirs(os.path.join(out_put_blocks, uscn))
    output_path = os.path.join(out_put_blocks, uscn)

    # Stack Bands
    #------------
    now_images = images_df.loc[images_df['id']==uscn]
    for band in band_order:
        try:
#             print(now_images.loc[now_images['band']==band]['path'].iloc[0])
            nowDs = gdal.Open(now_images.loc[now_images['band']==band]['path'].iloc[0])
            gt= nowDs.GetGeoTransform()
            proj= nowDs.GetProjection()
            windows=nowDs.ReadAsArray()
            windows=windows[np.newaxis,...]


            # Define tilting parameters
            # -------------------------
            rasXmax = nowDs.RasterXSize
            rasYmax = nowDs.RasterYSize
            block_size = 230       # Size of tile in pixels (X size Y size)

            overlap_perc = 0                    # Overlap of tiles

            overlap = 90#int(round(block_size*(overlap_perc/100),0))
            stepSize = block_size-overlap

            #print('Blocking ...')
            vrt_list = []
            #------------------------------------
            # Manage edge case
            #------------------------------------
            calc_x_steps = (rasXmax-overlap)/stepSize
            calc_y_steps = (rasYmax-overlap)/stepSize
            last_square =False
            if calc_y_steps != int(calc_y_steps):
                yminLim=0
                ymaxLim=block_size
                for y_step in range(int(calc_y_steps)):
                    nowBlockArr = windows[...,yminLim:ymaxLim,rasXmax-block_size:rasXmax]

                    yminLim = yminLim+stepSize
                    ymaxLim = ymaxLim+stepSize

                    nowGt = (gt[0] + ((rasXmax-block_size)*gt[1]), 
                              gt[1],0,
                              gt[3] - (int(y_step)*abs(stepSize*gt[5])),
                              0,gt[5])

                    out_file = os.path.join(output_path, uscn+'_'+band+'_BEN_'+str(y_step)+'_'+str(int(calc_x_steps)+1)+'.tif')
                    output_image(out_file,
                              nowGt, proj, nowBlockArr, gdal.GDT_UInt16)
                last_square=True

            if calc_x_steps !=int(calc_x_steps):
              # edge case - calculate additional blocks for y max
                xminLim=0
                xmaxLim=block_size
                for x_step in range(int(calc_x_steps)):

                    nowBlockArr = windows[..., rasYmax-block_size:rasYmax,xminLim:xmaxLim]
                    xminLim = xminLim+stepSize
                    xmaxLim = xmaxLim+stepSize

                    nowGt = (gt[0] + (int(x_step)*abs(stepSize*gt[1])), 
                              gt[1],0,
                              gt[3] - ((rasYmax-block_size)*abs(gt[5])),
                              0,gt[5])

                    out_file = os.path.join(output_path, uscn+'_'+band+'_BEN_'+str((int(calc_y_steps)+1))+'_'+str(x_step)+'.tif')
                    output_image(out_file,
                              nowGt, proj, nowBlockArr, gdal.GDT_UInt16)
                last_square=True

            #------------------------------------
            # Get Last square
            #------------------------------------
            if last_square:
                nowBlockArr = windows[..., rasYmax-block_size:rasYmax,rasXmax-block_size:rasXmax]

                nowGt = (gt[0] + ((rasXmax-block_size)*gt[1]), 
                          gt[1],0,
                          gt[3] - ((rasYmax-block_size)*abs(gt[5])),
                          0,gt[5])

                out_file = os.path.join(output_path, uscn+'_'+band+'_BEN_'+str((int(calc_y_steps)+1))+'_'+str((int(calc_x_steps)+1))+'.tif')
                output_image(out_file,
                          nowGt, proj, nowBlockArr, gdal.GDT_UInt16)

            #------------------------------------
            # Continue normal windows
            #------------------------------------
            windows = view_as_windows(windows,(windows.shape[0],block_size, block_size), step=stepSize)
            windows = windows[0]

            y_size = windows.shape[0]
            x_size = windows.shape[1]
            for y in tqdm(range(y_size)):
                for x in range(x_size):
                    y=y # For testing
                    x=x
                    nowBlockArr = windows[y][x]

                    nowGt = (gt[0] + (x*abs(stepSize*gt[1])), 
                              gt[1],0,
                              gt[3] - (y*abs(stepSize*gt[5])),
                              0,gt[5])    

                    out_file = os.path.join(output_path, uscn+'_'+band+'_BEN_'+str(y)+'_'+str(x)+'.tif')

                    output_image(out_file,
                              nowGt, proj, nowBlockArr, gdal.GDT_UInt16)
        except ValueError:
            print('Failed: ', now_images.loc[now_images['band']==band]['path'].iloc[0])
            break

lmodel = keras.models.load_model('saved_models/resnet50v2/')

###########################################
#STACK HERE
###########################################
'''
# For the scene - loop through unique grid ids (uni_grids)

for uni_gird in uni_grids:

# Red
redDs = gdal.Open(os.path.join(output_path, reconstruct_name+ (B02) +uni_gird+ '.tif'))
ImGT = redDs.GetGeoTransform()
ImPrj = redDs.GetProjection()
red_band = redDs.GetRasterBand(1)
red_arr = red_band.ReadAsArray()
redDs=None

#Green
greenDs = gdal.Open(os.path.join(output_path, reconstruct_name+ (B03) +uni_gird+ '.tif'))
green_arr = greenDs.ReadAsArray()
greenDs = None

#NIR
nirDs = gdal.Open(os.path.join(output_path, reconstruct_name+ (B04) +uni_gird+ '.tif'))
nir_arr = nirDs.ReadAsArray()
nirDs=None


nirDs = gdal.Open(os.path.join(output_path, reconstruct_name+ (B08) +uni_gird+ '.tif'))
nir_arr = nirDs.ReadAsArray()
nirDs=None

image_stack = np.stack((red_arr,green_arr,nir_arr))
output_img = /path/and/name
output_image(output_img, ImGT, ImPrj, image_stack, gdal.GDT_UInt16)
'''
ben_classes = ['coal', 'steel', 'other']

base_folder = 'deploy/deploy_blk/' 
for uni_scene in uni_scenes:
    if os.path.isfile(os.path.join(output_layers,'S2A_'+uni_scene + '_0_L2A',
                        uni_scene + '_BEN.gpkg')) or os.path.isfile(os.path.join(output_layers,
                        uni_scene + '_BEN.gpkg')) : 
        print('already done for ', uni_scene)
        continue
    
    print('For: ', uni_scene)

    bands_blocks = os.listdir(os.path.join(base_folder,uni_scene))
    bands_blocks_df = [[('_').join(x.split('_')[0:2]),
                        x.split('_')[2],
                        ('_').join(x.split('_')[-2:]),
                        os.path.join(base_folder, uni_scene, x)] for x in bands_blocks]
    bands_blocks_df = pd.DataFrame(bands_blocks_df, columns=['scene', 'band', 'block', 'path'])

    blocks=bands_blocks_df['block'].unique()
#     print(tqdm(blocks))
    probabilities_df=[]
    for block in blocks: 
        now_block=[]

        now_block.append(block[:-4])
        band_dict={}
        for band in band_order:
            nowBand_path = bands_blocks_df.loc[((bands_blocks_df['block'] == block) &\
                                                (bands_blocks_df['band'] == band))]['path'].iloc[0]
            nowBand = gdal.Open(nowBand_path)
            now_arr= nowBand.ReadAsArray()
            band_dict[band]=now_arr[np.newaxis,...]
#             print(band_dict.keys())
#             band_dict['Placeholder:0'] = np.array(False)
#             band_dict['BigEarthNet-19_labels_multi_hot']=np.expand_dims(np.zeros(19), 0)

            if band=='B04':
                geo_t = nowBand.GetGeoTransform()
                proj_now = nowBand.GetProjection()
                proj = osr.SpatialReference(wkt=proj_now)
                EPSG = int(proj.GetAttrValue('AUTHORITY',1))
                x_size = nowBand.RasterXSize
                y_size = nowBand.RasterYSize
                xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
                xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
                ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
                ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
            nowBand=None
        B02, B03, B04, B08 = band_dict['B02'][0], band_dict['B03'][0], band_dict['B04'][0], band_dict['B08'][0]
        B02 = (B02-ch1_mean)/ch1_std
        B03 = (B03-ch2_mean)/ch2_std
        B04 = (B04-ch3_mean)/ch3_std
        B08 = (B08-ch4_mean)/ch4_std
        features = np.array([ B02, B03, B04, B08]).transpose(1,2,0)
        probs = tf.nn.softmax(lmodel.predict(np.expand_dims(features,axis=0), verbose=0))
#       TF 2 MODEL PREDICT
#         probs= sess.run(model.probabilities, feed_dict=model.feed_dict(band_dict))
        for prob in probs[0]:
            now_block.append(prob.numpy())
        now_block.append(bounds_to_polygon([xmin, ymin, xmax, ymax]))
        probabilities_df.append(now_block)
    
    probabilities_df = gpd.GeoDataFrame(probabilities_df, columns=['block_id']+ben_classes+['geometry'])
    probabilities_df = probabilities_df.set_crs(EPSG, allow_override=True)
    probabilities_df.to_file(os.path.join(output_layers,
                        uni_scene + '_BEN.gpkg'), driver='GPKG')
    print('Done ', uni_scene + '_BEN.gpkg')
#     break 