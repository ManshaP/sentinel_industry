"""Bing Maps Tile handler and downloader

https://github.com/llgeek/Satellite-Aerial-Image-Retrieval

"""
import logging
import math
from itertools import chain
import re
import tempfile
from typing import Tuple
from io import BytesIO
import os
from pathlib import Path
import numpy as np
import shapely
import shapely.geometry
import rasterio
from PIL import Image
import requests
import sentinelhub as sh
import fiona
from fiona.crs import from_epsg
from osgeo import gdal
from shapely.geometry import box
import tqdm
import geojson

Image.MAX_IMAGE_PIXELS = 1000000000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server." + __name__)


class TileSystem:
    """Tile System Class to handle bing maps tiles in geographic,
    tile, and pixel coordinates"""

    MINLAT, MAXLAT = -85.05112878, 85.05112878
    MINLON, MAXLON = -180.0, 180.0

    @staticmethod
    def clip(val: float, minval: float, maxval: float) -> float:
        """Clips a number to be specified minval and maxval values.

        Arguments:
            val (float): value to be clipped
            minval (float): minimal value bound
            maxval (float): maximal value bound

        Returns:
            float: clipped value
        """
        return min(max(val, minval), maxval)

    @staticmethod
    def map_size(level: int) -> int:
        """Determines the map width and height (in pixels) at a specified level

        Arguments:
            level (int): level of detail, from 1 (lowest detail) to 23 (highest detail)
        Returns:
            int: The map width and height in pixels (width == height)
        """
        return 256 << level  # bitwise shift operator

    @staticmethod
    def latlong_to_pixelxy(lat: float, long: float, level: int) -> Tuple[int, int]:
        """Converts a point from latitude/longitude WGS-84 coordinates (in degrees)
        into pixel XY coordinates a specified level of detail

        Arguments:
            lat (float): Latitude of the point, in degrees
            long (float): Longitude of the point, in degrees
            level (int): Level of detail, from 1 (lowest detail) to 23 (highest detail)

        Returns:
            [int, int] -- [X coordinates in pixels; Y coordinates in pixels]
        """
        lat = TileSystem.clip(lat, TileSystem.MINLAT, TileSystem.MAXLAT)
        long = TileSystem.clip(long, TileSystem.MINLON, TileSystem.MAXLON)

        x = (long + 180) / 360
        sinlat = math.sin(lat * math.pi / 180)
        y = 0.5 - math.log((1 + sinlat) / (1 - sinlat)) / (4 * math.pi)

        mapsize = TileSystem.map_size(level)
        pixel_x, pixel_y = math.floor(
            TileSystem.clip(x * mapsize + 0.5, 0, mapsize - 1)
        ), math.floor(TileSystem.clip(y * mapsize + 0.5, 0, mapsize - 1))
        return pixel_x, pixel_y

    @staticmethod
    def pixelxy_to_latlong(
        pixel_x: int, pixel_y: int, level: int
    ) -> Tuple[float, float]:
        """Converts a pixel from pixel XY coordinates at a specified level of detail
        into latitude/longitude WGS-84 coordinates (in degrees)

        Arguments:
            pixel_x (int): X coordinate of the point, in pixels
            pixel_y (int): Y coordinate of the point, in pixels
            level (int): Level of detail, from 1 (lowest detail) to 23 (highest detail)

        Returns:
            float, float: Latitude in degrees; Longitude in degrees
        """
        mapsize = TileSystem.map_size(level)
        x = TileSystem.clip(pixel_x, 0, mapsize - 1) / mapsize - 0.5
        y = 0.5 - 360 * TileSystem.clip(pixel_y, 0, mapsize - 1) / mapsize
        lat = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
        lon = 360 * x
        return lat, lon

    @staticmethod
    def pixelxy_to_tilexy(pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Converts pixel XY coordinates into tile XY coordinates
        of the tile containing the specified pixel.

        Arguments:
            pixel_x (int): Pixel X coordinate
            pixel_y (int): Pixel Y coordinate

        Returns:
            int, int: Tile X coordinate; Tile Y coordinate
        """
        return math.floor(pixel_x / 256), math.floor(pixel_y / 256)

    @staticmethod
    def tilexy_to_pixelxy(tile_x: int, tile_y: int) -> Tuple[int, int]:
        """Converts tile XY coordinates into pixel XY coordinates
        of the upper-left pixel of the specified tile

        Arguments:
            tile_x (int): Tile X coordinate
            tile_y (int): Tile Y coordinate

        Returns:
            int, int: pixel X coordinate; pixel Y coordinate
        """
        return tile_x * 256, tile_y * 256

    @staticmethod
    def tilexy_to_quadkey(tile_x: int, tile_y: int, level: int) -> str:
        """Converts tile XY coordinates into a QuadKey at a specified
        level of detail interleaving tile_y with tile_x

        Arguments:
            tile_x (int): Tile X coordinate
            tile_y (int): Tile Y coordinate
            level (int): Level of detail, from 1 (lowest detail) to 23 (highest detail)

        Returns:
            [string] -- [A string containing the QuadKey]
        """
        tile_x_bits = "{0:0{1}b}".format(tile_x, level)
        tile_y_bits = "{0:0{1}b}".format(tile_y, level)
        quadkeybinary = "".join(chain(*zip(tile_y_bits, tile_x_bits)))
        return "".join([str(int(num, 2)) for num in re.findall("..?", quadkeybinary)])

    @staticmethod
    def quadkey_to_tile_xy(quadkey: str) -> Tuple[int, int]:
        """Converts a QuadKey into tile XY coordinate

        Arguments:
            quadkey (string): QuadKey of the tile

        Returns:
            int, int: Tile X coordinate; Tile Y coordinate
        """
        quadkeybinary = "".join(["{0:02b}".format(int(num)) for num in quadkey])
        tile_x, tile_y = int(quadkeybinary[1::2], 2), int(quadkeybinary[::2], 2)
        return tile_x, tile_y


class AerialImageRetrieval:
    """The class for aerial image retrieval
    To create an AerialImageRetrieval object, simply give upper left latitude, longitude,
    and lower right latitude and longitude
    """

    def __init__(self, aoi_geometry: dict, level: int, cache_dir: str):
        """init

        Args:
            aoi_geometry (dict): aoi geometry
            level (int): zoom level
        """
        self.aoi_geometry = aoi_geometry
        self.level = level
        self.lat1, self.lon1, self.lat2, self.lon2 = self._get_bounds_minmax(
            aoi_geometry
        )
        self.bingmaps_key = (
            "Ao_F5t1lNNVkXyTmqQFgtSPGwo7vrpdCeK37ETjxjhoffYazM7AIT9LFcSWT7JYy"
        )
        self.tileurl, self.subdomain, self.tilesize = self._get_tileurl()
        self.image = None
        self.cache_dir = cache_dir

    def _get_tileurl(self) -> Tuple[str, str, int]:
        """Derive tileurl from request to imagery service using bingmaps key

        Returns:
            Tuple[str, str, int]: tileurl, subdomain, tilesize
        """
        baseurl = (
            f"http://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial?output=json&"
            f"include=ImageryProviders&key={self.bingmaps_key}"
        )
        resp = requests.get(baseurl)
        resp_json = resp.json()["resourceSets"][0]["resources"][0]
        logger.info(resp_json)
        tileurl = resp_json["imageUrl"]
        subdomain = resp_json["imageUrlSubdomains"][
            1
        ]  # TODO check if it should be changed every new request
        tilesize = resp_json["imageWidth"]
        self.imageUrlSubdomains = resp_json["imageUrlSubdomains"]
        return tileurl, subdomain, tilesize

    def _get_bounds_minmax(
        self, aoi_geometry: dict
    ) -> Tuple[float, float, float, float]:
        """Calculate min max bounds of input aoi geometry"""
        aoi_geometry = shapely.geometry.shape(aoi_geometry)
        lon1, lat2, lon2, lat1 = aoi_geometry.bounds
        return lat1, lon1, lat2, lon2

    def download_image(self, subdomain: str, quadkey: str, cache_dir: str):
        """This method is used to download a tile image given the quadkey
        from Bing tile system

        Arguments:
            quadkey (string): The quadkey for a tile image

        Returns:
            Image: A PIL Image
        """
        # choosen_subdomain = random.choice(self.imageUrlSubdomains)
        tileurl = self.tileurl.format(subdomain=subdomain, quadkey=quadkey)
        # check if file exists
        cache_file = self.cache_file_exists(url=tileurl, cache_dir=cache_dir)
        if cache_file[0]:
            return Image.open(cache_file[1])
        else:
            r = requests.get(tileurl, timeout=10)
            # logger.info(tileurl)
            if r.status_code == 200:
                self.save_image(cache_dir=cache_dir, image_response=r, url=tileurl)
                # time.sleep(0.001)
                return Image.open(BytesIO(r.content))
            else:
                print(r.status_code)
                print(r.text)

    def cache_file_exists(self, url, cache_dir):
        """Check if tile exists in cache

        Args:
            url (str): url
            cache_dir (str): cache directory

        Returns:
            bool, str: true, path to tile
        """
        fn = self.generate_caching_name(url)
        fn = f"{fn}.jpeg"
        fn_path = Path(cache_dir, fn)
        if os.path.exists(fn_path):
            return True, fn_path
        else:
            return False, None

    def generate_caching_name(self, url: str) -> str:
        """
        _summary_

        Args:
            url (str): _description_

        Returns:
            str: _description_
        """
        x = Path(url)
        x = x.name
        x = x.replace(".", "_")
        x = x.replace("?", "_")
        x = x.replace("=", "_")
        return x

    def save_image(self, cache_dir: str, image_response: str, url: str):
        """
        _summary_

        Args:
            cache_dir (str): _description_
            image_response (str): _description_
        """
        img = Image.open(BytesIO(image_response.content))
        dst_fn = self.generate_caching_name(url)
        dst_fn = f"{dst_fn}.jpeg"
        dst_path = Path(cache_dir, dst_fn)
        img.save(dst_path.as_posix()) #.convert('RGB')

    def horizontal_retrieval_and_stitch_image(
        self, tile_x_start: int, tile_x_end: int, tile_y: int, level: int
    ) -> Tuple[bool, Image.Image]:
        """Horizontally retrieve tile images and then stitch them together,
        start from tile_x_start and end at tile_x_end, tile_y will remain the same

        Arguments:
            tile_x_start (int): the starting tile_x index
            tile_x_end (int): the ending tile_x index
            tile_y (int): the tile_y index
            level (int): level used to retrieve image

        Returns:
            boolean, Image: whether such retrieval is successful;
                If successful, returning the stitched image, otherwise None
        """
        count = 0
        max_usage = 1
        domain_idx = 0

        imagelist = []
        logger.info(f"len horizontal list: {tile_x_end-tile_x_start}")
        for tile_x in range(tile_x_start, tile_x_end + 1):
            if count > max_usage:
                domain_idx += 1
                if domain_idx == len(self.imageUrlSubdomains):
                    domain_idx = 0
                count = 0
            selected_subdomain = self.imageUrlSubdomains[domain_idx]
            count += 1
            quadkey = TileSystem.tilexy_to_quadkey(tile_x, tile_y, level)
            image = self.download_image(
                selected_subdomain, quadkey, cache_dir=self.cache_dir
            )
            imagelist.append(image)
        result = Image.new("RGB", (len(imagelist) * self.tilesize, self.tilesize))
        for i, image in enumerate(imagelist):
            result.paste(image, (i * self.tilesize, 0))
        return True, result

    def imagery_retrieval(self):
        """The main aerial retrieval method
        It will firstly determine the appropriate level used to retrieve the image.
        The appropriate level should satisfy:
            1. All the tile image within the given bounding box at that level
                should all exist
            2. The retrieved image cannot exceed the maximum supported image
                size, which is 8192*8192

        Then for the given level, we can download each aerial tile image,
        and stitch them together.
        Lastly, we have to crop the image based on the given bounding box
        Returns:
            Image: image
        """
        pixel_x1, pixel_y1 = TileSystem.latlong_to_pixelxy(
            self.lat1, self.lon1, self.level
        )
        pixel_x2, pixel_y2 = TileSystem.latlong_to_pixelxy(
            self.lat2, self.lon2, self.level
        )

        pixel_x1, pixel_x2 = min(pixel_x1, pixel_x2), max(pixel_x1, pixel_x2)
        pixel_y1, pixel_y2 = min(pixel_y1, pixel_y2), max(pixel_y1, pixel_y2)

        if abs(pixel_x1 - pixel_x2) <= 1 or abs(pixel_y1 - pixel_y2) <= 1:
            print("Cannot find a valid aerial imagery for the given bounding box!")
            return None

        tile_x1, tile_y1 = TileSystem.pixelxy_to_tilexy(pixel_x1, pixel_y1)
        tile_x2, tile_y2 = TileSystem.pixelxy_to_tilexy(pixel_x2, pixel_y2)

        # Stitch the tile images together
        result = Image.new(
            "RGB",
            (
                (tile_x2 - tile_x1 + 1) * self.tilesize,
                (tile_y2 - tile_y1 + 1) * self.tilesize,
            ),
        )
        retrieve_sucess = False
        logger.info("Download..")
        for tile_y in tqdm.tqdm(range(tile_y1, tile_y2 + 1)):
            (
                retrieve_sucess,
                horizontal_image,
            ) = self.horizontal_retrieval_and_stitch_image(
                tile_x1, tile_x2, tile_y, self.level
            )
            if not retrieve_sucess:
                break
            result.paste(horizontal_image, (0, (tile_y - tile_y1) * self.tilesize))
            # time.sleep(2)
        logger.info("Crop..")
        # Crop the image based on the given bounding box
        leftup_corner_x, leftup_corner_y = TileSystem.tilexy_to_pixelxy(
            tile_x1, tile_y1
        )
        self.image = result.crop(
            (
                pixel_x1 - leftup_corner_x,
                pixel_y1 - leftup_corner_y,
                pixel_x2 - leftup_corner_x,
                pixel_y2 - leftup_corner_y,
            )
        )
        return self.image

    def georeference_and_save(self, dst_path: str) -> str:
        """Georeference and save image as geotiff

        Args:
            dst_path (str): destination path to save image

        Returns:
            str: destination path
        """
        array = np.array(self.image)
        array = array.astype("uint8")
        transform = rasterio.transform.from_bounds(
            west=self.lon1,
            south=self.lat2,
            east=self.lon2,
            north=self.lat1,
            width=array.shape[1],
            height=array.shape[0],
        )
        out_meta = {}
        out_meta["driver"] = "GTiff"
        out_meta["height"] = array.shape[0]
        out_meta["width"] = array.shape[1]
        out_meta["count"] = array.shape[2]
        out_meta["transform"] = transform
        out_meta["crs"] = rasterio.crs.CRS.from_epsg(4326)
        out_meta["dtype"] = array.dtype

        array = np.moveaxis(array, -1, 0)
        with rasterio.open(dst_path, "w", **out_meta) as dest:
            dest.write(array)
        return dst_path


# def mp_download_image(input):
#     tile_x = input[0]
#     tile_y= input[1]
#     level= input[2]
#     tileurl= input[3]
#     subdomains= input[4]
#     quadkey = TileSystem.tilexy_to_quadkey(tile_x, tile_y, level)
#     subdomain = random.choice(subdomains)
#     tileurl = tileurl.format(subdomain=subdomain, quadkey=quadkey)
#     with urllib.request.urlopen(tileurl) as file:
#         return Image.open(file)


def clip_to_aoi(aoi_geometry: dict, src_path: str, dst_path: str) -> str:
    """Clip image to specified aoi.

    Args:
        aoi_geometry (dict): aoi geometry
        src_path (str): source image path
        dst_path (str): destination image path

    Returns:
        str: destination image path
    """
    with tempfile.TemporaryDirectory() as tempdir:
        shp_file = f"{tempdir}/geometry.shp"
        schema = {"geometry": "Polygon", "properties": {"id": "int"}}
        with fiona.open(
            shp_file, "w", "ESRI Shapefile", schema, crs=from_epsg(4326)
        ) as layer:
            layer.write({"geometry": aoi_geometry, "properties": {"id": 1}})
        local_file = gdal.Warp(
            destNameOrDestDS=dst_path,
            srcDSOrSrcDSTab=src_path,
            cutlineDSName=shp_file,
            dstNodata=0,
            cropToCutline=True,
        )
    local_file = None
    del local_file
    return dst_path


def change_resolution(src_image: str, dst_image: str, resolution=0.5) -> str:
    """
    Change raster resolution to a set value and return in source projection.

    Args:
        src_image (str): path to source image.
        dst_image (str): path to target image.
        resolution (float, optional): Spatial resolution in meter.
            Defaults to 0.5.

    Returns:
        str: path to created target image.
    """
    with rasterio.open(src_image) as rs_ds:
        src_crs = rs_ds.crs
        if src_crs.to_epsg() == 4326:
            bbox = rs_ds.bounds
            bounding_box = box(*bbox)
            centroid = bounding_box.centroid
            _x, _y = [(x[0], x[1]) for x in centroid.coords][0]
            crs_target = sh.geo_utils.get_utm_crs(_x, _y)
        else:
            crs_target = src_crs
    image_scaled = gdal.Warp(
        destNameOrDestDS=dst_image,
        srcDSOrSrcDSTab=src_image,
        dstSRS=crs_target,
        xRes=resolution,
        yRes=resolution,
    )
    image_scaled.FlushCache()
    image_scaled = None
    # change intermediate result back to source projection
    # target_file = gdal.Warp(destNameOrDestDS=dst_image,
    #                        srcDSOrSrcDSTab=image_scaled,
    #                        dstSRS=src_crs,
    #                        resampleAlg='bilinear')
    # target_file.FlushCache()
    # target_file = None
    return dst_image


def download_bing_aoi(
    geojson_pth: str,
    zoom_level: int,
    target_dir: str,
    bing_cache: str,
    delete_intermediate: bool,
):
    """
    _summary_

    Args:
        geojson_pth (str): _description_
        zoom_level (int): _description_
        target_dir (str): _description_
        bing_cache (str): _description_
        delete_intermediate (bool): _description_

    Returns:
        _type_: _description_
    """

    aoi_file = Path(geojson_pth)
    scaled_file = Path(f"{target_dir}/{aoi_file.stem}_{zoom_level}_1m_utm.tif")
    with open(aoi_file) as f:
        feat_collection = geojson.load(f)
    print(feat_collection["features"][0]["geometry"])
    geom = feat_collection["features"][0]["geometry"]

    download_file = str(Path(target_dir, f"bingmaps_{aoi_file.stem}_{zoom_level}.tif"))
#     print(download_file)
    imgretrieval = AerialImageRetrieval(
        aoi_geometry=geom, level=zoom_level, cache_dir=bing_cache
    )
#     print('AR done')
    img = imgretrieval.imagery_retrieval()
    _ = imgretrieval.georeference_and_save(dst_path=download_file)

    img_scaled_res = change_resolution(
        src_image=download_file, dst_image=scaled_file.as_posix(), resolution=1.
    )

    # clipped = str(Path(target_dir, f"bingmaps_{aoi_file.stem}_lvl{zoom_level}_50cm.tif"))
    # img_clipped_path = clip_to_aoi(aoi_geometry=geom,
    #                                src_path=download_file,
    #                                dst_path=clipped)

    if delete_intermediate:
        os.remove(download_file)
    return True


# if __name__ == "__main__":
#     from pathlib import Path
#     import geojson

#     aoi_file = Path("./data/de_nandlstadt.geojson")
#     zoom_level = 18
#     target_dir = "./data/"
#     cache_dir = "./data/bing_cache/"

#     download_bing_aoi(
#         geojson_pth=aoi_file,
#         zoom_level=zoom_level,
#         target_dir=target_dir,
#         bing_cache=cache_dir,
#         delete_intermediate=True,
#     )
