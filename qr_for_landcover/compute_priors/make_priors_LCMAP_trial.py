import numpy as np
import pandas as pd
import rasterio
import sys
import os
from importlib import reload
import time
import rasterio
from rasterio.windows import Window
import torch as T
from torchvision.transforms import GaussianBlur

here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(here, "..", "scripts"))
import landcover_definitions as lc

cyverse = True

if cyverse:
    use_gpu = True
    pairs_path = '/data-store/iplant/home/emilynagamoto/NAIP_LCMAP/'
else: 
    use_gpu = False
    pairs_path = 'C:/Users/emna9376/OneDrive - UCB-O365/CIRES_IRP/naip_lcmap_pairs/'
    
if use_gpu:
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
else:
    device = T.device('cpu')
    
sys.path.append('../scripts')
import util
from general_prior_calculation import calculate_prior_preblurred

# CC_data_dir = os.environ.get(
#     "CC_DATA_DIR",
#     os.path.join(here, "..", "..", "..", "torchgeo_data",
#                  "cvpr_chesapeake_landcover", "cvpr_chesapeake_landcover"))
# cooc_data_dir = os.path.join(here, "cooccurrence_matrices", "chesapeake")

# def unique_tile_ids_from_dir(data_dir_this):
#     """Get tile ids from a provided data dir."""
#     return np.unique([x[2:17] for x in os.listdir(data_dir_this)])

# def condense_cc7_to_cc5_cooccurrance(cooc_matrix_in):
#     """Combine Impervious classes."""
#     cooc = cooc_matrix_in.copy()
#     cooc[4] = cooc[4:].sum(axis=0)
#     return cooc[:5]
# Copied from https://www.usgs.gov/special-topics/lcmap/lcmap-class-legend-and-description
LCMAP_CLASS_DEFINITIONS = {
    0: ("No Data", "No data value"),
    1: ("Developed", "Built-up land such as buildings, roads, and other constructed surfaces."),
    2: ("Cropland", "Land used to produce food, fiber, and fuel, whether vegetated or bare at the time."),
    3: ("Grass/Shrub", "Land dominated by grasses, forbs, or shrubs: at least 10% grass/shrub cover and under 10% tree cover."),
    4: ("Tree Cover", "Tree-covered land with tree cover density above 10%."),
    5: ("Water", "Streams, canals, lakes, reservoirs, bays, or ocean."),
    6: ("Wetland", "Land where water saturation controls soil, vegetation, and animal communities; a mosaic of water, bare soil, and herbaceous or woody cover."),
    7: ("Ice/Snow", "Perennial ice and snow that does not fully melt in summer."),
    8: ("Barren", "Natural soil, sand, or rock with less than 10% vegetation."),
}

# Copied from the embedded color table in the lCMAP files (above)
LCMAP_CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (255, 50, 50, 255),     # Developed
    2: (190, 140, 90, 255),    # Cropland
    3: (230, 240, 210, 255),   # Grass/Shrub
    4: (28, 99, 48, 255),      # Tree Cover
    5: (0, 112, 255, 255),     # Water
    6: (179, 217, 255, 255),   # Wetland
    7: (255, 255, 255, 255),   # Ice/Snow
    8: (179, 174, 163, 255),   # Barren
}

lcmap_codes = [1, 2, 3, 4, 5, 6, 7, 8]
lcmap_names = ["Developed", "Cropland", "Grass/Shrub", "Tree Cover",
               "Water", "Wetland", "Ice/Snow", "Barren"]

targets = ["Developed", "Cropland", "Grass/Shrub", "Tree Cover",
               "Water", "Wetland", "Ice/Snow", "Barren"]  

# rows = targets, cols = LCMAP classes
## first pass just use basic for the same classes where it is 90% the same class but then some fuzziness for the 10%
recipes = {
    #                 develop  crop  grass  tree  water  wetland  ice  barren
    "Developed":      [0.90,  0.05,  0.00,  0.00,  0.00,  0.00,  0.00,  0.05],
    "Cropland":       [0.00,  0.90,  0.05,  0.05,  0.00,  0.00,  0.00,  0.00],
    "Grass/Shrub":    [0.00,  0.05,  0.90,  0.05,  0.00,  0.00,  0.00,  0.00],
    "Tree Cover":     [0.00,  0.05,  0.50,  0.90,  0.00,  0.00,  0.00,  0.00],
    "Water":          [0.00,  0.00,  0.00,  0.00,  0.90,  0.05,  0.00,  0.05],
    "Wetland":        [0.00,  0.00,  0.05,  0.00,  0.05,  0.90,  0.00,  0.00],
    "Ice/Snow":       [0.00,  0.05,  0.00,  0.00,  0.00,  0.00,  0.90,  0.05],
    "Barren":         [0.00,  0.05,  0.05,  0.00,  0.00,  0.00,  0.00,  0.90],
}

def create_map_raw_lc_to_idx(class_definitions):
    LC_TO_IDX = np.zeros(np.array(list(class_definitions.keys())).max() + 1, dtype=np.uint8)
    for i, k in enumerate(class_definitions.keys()):
        LC_TO_IDX[k] = i
    return LC_TO_IDX  

def get_colors(class_colors):
    return np.array([class_colors[c] for c in class_colors.keys()]) / 255.0

def compile_and_save_one_hot_lcmap_cc(lcmap_fp, remap_to_idx=True):
    # this creates lookup table for lcmap codes and positions 
    if remap_to_idx: 
        lcmap = create_map_raw_lc_to_idx(LCMAP_CLASS_DEFINITIONS)[rasterio.open(lcmap_fp).read()[0]]
    else:
        lcmap = rasterio.open(lcmap_fp).read()[0]
        
    num_classes = len(get_colors(LCMAP_CLASS_COLORS))
    lcmap_onehot = np.zeros((num_classes, lcmap.shape[0],lcmap.shape[1]))
    for c in range(num_classes):
        lcmap_onehot[c][lcmap == c] = 1

    prof_out = rasterio.open(lcmap_fp).profile.copy()
    prof_out['count'] = num_classes
    
    lcmap_fp_out = lcmap_fp.replace('lcmap.', 'lcmap_onehot.')
    
    with rasterio.open(lcmap_fp_out, 'w', **prof_out) as f:
        f.write(lcmap_onehot)

        

def compile_and_save_one_blurred_hot_lcmap_cc(lcmap_onehot_fp,
                                              blur_kernel_size=101,
                                              blur_sigma=51,
                                              device=T.device('cuda:0')):
    
    blur = GaussianBlur(blur_kernel_size, blur_sigma)

    with rasterio.open(lcmap_onehot_fp) as src:
        prof_out = src.profile.copy()
        data = src.read()                      # (C, H, W)

    out = np.empty(data.shape, dtype=np.uint8)
    with T.no_grad():
        for c in range(data.shape[0]):
            ch = T.from_numpy(data[c:c+1]).to(device).float()
            ch = blur(ch).clamp_(0, 1)
            out[c] = (ch * 255.).byte().cpu().numpy()[0]
            del ch
    T.cuda.empty_cache()

    lcmap_fp_out = lcmap_onehot_fp.replace(
        'lcmap_onehot',
        f'lcmap_onehot_blurred_kernelsize_{blur_kernel_size}_sigma_{blur_sigma}')
    prof_out.update(dtype='uint8')
    with rasterio.open(lcmap_fp_out, 'w', **prof_out) as f:
        f.write((out).astype(np.uint8))

        
def make_and_save_priors_cc(tiles, 
                            dir_this,
                            version, 
                            cooc_matrix,
                            plot_mapping_matrix=False):

    # by default use OSM
    ignore_osm = False
    ignore_buildings = False
    spatial_smooth = True
    
    #these index values are what they are for LCMAP
    road_idx_cc = 1
    building_idx_cc = 1
    water_idx_cc = 5
        
    if version == 'from_cooccurrences_101_31_no_osm_no_buildings':
        lcmap_blur_kernelsize = 101
        lcmap_blur_sigma = 31
        ignore_osm = True
        ignore_buildings = True
            
    else:
        print(f'version {version} not understood')
        return
    
    print(version)
    if spatial_smooth: print(lcmap_blur_sigma)
    else: print("not performing spatial smooth")
        
        
    if plot_mapping_matrix:
        plt.figure()
        plt.imshow(nlcd_mapping_matrix)
        plt.title('mapping matrix being used');
        plt.show()
        
    for i,tile_id in enumerate(tiles):
        print(i, tile_id)
        t1 = time.time()

        if spatial_smooth:
            # note this assumes that you've made this blurred input already.
            lcmap_fp = f'{dir_this}/{tile_id}_lcmap_onehot_blurred_kernelsize_' + \
                            f'{lcmap_blur_kernelsize}_sigma_{lcmap_blur_sigma}.tif'
            
        else:
            lcmap_fp = f'{dir_this}/{tile_id}_lcmap_onehot.tif'
        print(lcmap_fp)
        
        #by default these will get ignored
        roads_fp = f'{dir_this}/m_{tile_id}_roads.tif'
        buildings_fp = f'{dir_this}/m_{tile_id}_buildings.tif'
        waterways_fp = f'{dir_this}/m_{tile_id}_waterways.tif'
        waterbodies_fp = f'{dir_this}/m_{tile_id}_waterbodies.tif'
        
        if not ignore_buildings:
            buildings = rasterio.open(buildings_fp).read()[0]
            
        if not ignore_osm:
            roads = rasterio.open(roads_fp).read()[0]
            waterways = rasterio.open(waterways_fp).read()[0]
            waterbodies = rasterio.open(waterbodies_fp).read()[0]
            water = np.maximum(waterways, waterbodies)
            water = np.maximum(waterbodies, util.blur(waterways,waterways_preblur_radius)==1)

        # blur water
        if not ignore_osm:
            roads = util.bag_from_numpy(roads.astype('float'),radius=roads_bag_radius)
            water = util.bag_from_numpy(water.astype('float'),radius=water_bag_radius)
            
        if not ignore_buildings:
            buildings = util.bag_from_numpy(buildings.astype('float'),radius=buildings_bag_radius)

        win = Window(col_off=0, row_off=0, width=2000, height=2000)
        sl = win.toslices()
        
        with rasterio.open(lcmap_fp) as lcmap_preblurred:
            blurred = lcmap_preblurred.read(window=win)
            prof_out = lcmap_preblurred.profile.copy()
            prof_out.update(height=int(win.height), width=int(win.width),
                    transform=lcmap_preblurred.window_transform(win))
            shape = (int(win.height),int(win.width))
            
            if ignore_buildings:
                print('ignoring buildings')
                buildings_win = np.zeros(shape, dtype=np.float32)
            
            if ignore_osm:
                print('ignoring roads, and water')
                roads_win = np.zeros(shape, dtype=np.float32)
                water_win = np.zeros(shape, dtype=np.float32)

            prior_this_5 = calculate_prior_preblurred(blurred,
                                              roads_win, buildings_win, water_win,
                                              cooc_matrix, road_idx=road_idx_cc,
                                              building_idx=building_idx_cc,
                                              water_idx=water_idx_cc)
            # using the same script from previous - general_prior_calculation.py
            # prof_out = lcmap_preblurred.profile.copy()
            # prior_this_5 = calculate_prior_preblurred(lcmap_preblurred.read(),
            #                                     roads,
            #                                     buildings,
            #                                     water,
            #                                     cooc_matrix,
            #                                     road_idx = road_idx_cc,
            #                                     building_idx = building_idx_cc,
            #                                     water_idx = water_idx_cc)
        
        # now exclude nodata classes from the prior
        prior_this = prior_this_5[1:]
        
        # write this prior
        prof_out['count'] = prior_this.shape[0]
        with rasterio.open(f'{dir_this}/{tile_id}_prior_{version}.tif', 'w', **prof_out) as f:
            f.write((prior_this*255.).astype(np.uint8))

        t2 = time.time()
        print(t2-t1, ' seconds')



    
     
if __name__ == "__main__":
    M = pd.DataFrame(recipes, index=targets)[lcmap_names]   
    M.columns = lcmap_codes 
    cooc_matrix_file1 = M.to_numpy()
    
    ########## STEP 1
    lcmap_naip_dict = {'hayman':'hayman_co_m_3910554_se_13_060_20190913_20191121'}

    # load one pair
    file1_firename = 'hayman'
    #lcmap_file1 = pairs_path +  'hayman_lcmap.tif'
    #naip_file1 = pairs_path + 'hayman_co_m_3910554_se_13_060_20190913_20191121_image.tif'
    #lcmap_fp = f'{pairs_path}{file1_firename}_lcmap.tif'

    tiles = [file1_firename]

    make_and_save_priors_cc(tiles,  
                        pairs_path, 
                        'from_cooccurrences_101_31_no_osm_no_buildings',  
                        cooc_matrix_file1,
                        plot_mapping_matrix=False)
        