# coding=utf-8

import osr, ogr
import gdal
import numpy as np
import os


def raster2array(rasterfn):
    '''
    create array from raster
    Agrs:
        rasterfn: tiff file path
    Returns:
        array: tiff data, an 2D array
    '''
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    array = np.asarray(array)
    del raster
    return array,originX,originY,pixelWidth,pixelHeight



def array2raster_GDT_Byte(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = longitude_start
    originY = latitude_start
    # open geotiff
    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(newRasterfn):
        os.remove(newRasterfn)
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    # Add Color Table
    # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # Write Date to geotiff
    outband = outRaster.GetRasterBand(1)
    ndv = 255
    outband.SetNoDataValue(ndv)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # Close Geotiff
    outband.FlushCache()
    del outRaster

def array2raster(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = longitude_start
    originY = latitude_start
    # open geotiff
    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(newRasterfn):
        os.remove(newRasterfn)
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    # Add Color Table
    # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # Write Date to geotiff
    outband = outRaster.GetRasterBand(1)

    outband.SetNoDataValue(ndv)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # Close Geotiff
    outband.FlushCache()
    del outRaster



def array2raster_polar(newRasterfn,longitude_start,latitude_start,pixelWidth,pixelHeight,array,ndv = -999999):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = longitude_start
    originY = latitude_start
    # open geotiff
    driver = gdal.GetDriverByName('GTiff')
    if os.path.exists(newRasterfn):
        os.remove(newRasterfn)
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    # Add Color Table
    # outRaster.GetRasterBand(1).SetRasterColorTable(ct)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    # Write Date to geotiff
    outband = outRaster.GetRasterBand(1)





    outband.SetNoDataValue(ndv)
    outband.WriteArray(array)
    # outRasterSRS.ImportFromEPSG(4326)
    # outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # ref = osr.SpatialReference()
    # outRasterSRS = osr.SpatialReference()
    # ref_chr = r"PROJCS[\"NSIDC EASE-Grid North\",GEOGCS[\"Unspecified datum based upon the International 1924 Authalic Sphere\",DATUM[\"Not_specified_based_on_International_1924_Authalic_Sphere\",SPHEROID[\"International 1924 Authalic Sphere\",6371228,0,AUTHORITY[\"EPSG\",\"7057\"]],TOWGS84[-9036842.762,25067.525,0,9036842.763000002,0,-25067.525],AUTHORITY[\"EPSG\",\"6053\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4053\"]],PROJECTION[\"Lambert_Azimuthal_Equal_Area\"],PARAMETER[\"latitude_of_center\",90],PARAMETER[\"longitude_of_center\",0],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],AUTHORITY[\"EPSG\",\"3408\"]]"
    # ref_chr = r'PROJCS["Grenada 1953 / British West Indies Grid",GEOGCS["Grenada 1953",DATUM["Grenada_1953",SPHEROID["Clarke 1880 (RGS)",6378249.145,293.465,AUTHORITY["EPSG","7012"]],TOWGS84[72,213.7,93,0,0,0,0],AUTHORITY["EPSG","6603"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4603"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-62],PARAMETER["scale_factor",0.9995],PARAMETER["false_easting",400000],PARAMETER["false_northing",0],AUTHORITY["EPSG","2003"],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    # ref_chr = r'PROJCS["North_Pole_Lambert_Azimuthal_Equal_Area",GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["False_Easting",0],PARAMETER["False_Northing",0],PARAMETER["Central_Meridian",0],PARAMETER["Latitude_Of_Origin",90],UNIT["Meter",1],AUTHORITY["EPSG","9122"]]'
    # ref_chr = "PROJCS['North_Pole_Lambert_Azimuthal_Equal_Area',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Azimuthal_Equal_Area'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Latitude_Of_Origin',90.0],UNIT['Meter',1.0]]"
    # ref_chr = ref_chr.replace('\\','')
    # print ref_chr
    # "PROJCS['North_Pole_Lambert_Azimuthal_Equal_Area',GEOGCS['GCS_WGS_1984',DATUM['D_WGS_1984',SPHEROID['WGS_1984',6378137.0,298.257223563]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Lambert_Azimuthal_Equal_Area'],PARAMETER['False_Easting',0.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',0.0],PARAMETER['Latitude_Of_Origin',90.0],UNIT['Meter',1.0]]"
    # outRasterSRS.ImportFromWkt(ref_chr)
    # print outRasterSRS
    # outRaster.SetProjection(outRasterSRS.ExportToWkt())
    # outRaster.SetProjection(ref_chr)
    outband.FlushCache()
    del outRaster


if __name__ == '__main__':
    pass

