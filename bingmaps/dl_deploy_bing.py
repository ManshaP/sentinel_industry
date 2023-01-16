import glob
import os


aoi_files= glob.glob('/home/users/pete_nut/sentinel_industry/deploy/cl_look_polygons/*.geojson')

from bingmaps_downloader import download_bing_aoi
for aoi_file in aoi_files:
    download_bing_aoi(
                geojson_pth=aoi_file,
                zoom_level=17,
                target_dir = "/gws/nopw/j04/aopp/manshausen/bing_dl/deploy/",
                bing_cache = "/gws/nopw/j04/aopp/manshausen/bing_dl/deploy/",
                delete_intermediate=True,)
    for d in glob.glob("/gws/nopw/j04/aopp/manshausen/bing_dl/deploy/*.jpeg"):
        os.remove(d)