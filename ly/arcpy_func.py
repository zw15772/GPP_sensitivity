# coding=gbk

import os
import arcpy
from arcpy.sa import *
import log_process
import time
import sys

this_root = 'd:\\ly\\project06\\'
# this_root = 'D:/project06/'

def mk_dir( dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)



class Func():

    def __init__(self):
        pass

    def resample(self,in_dir,out_dir):
        # ndvi_8km_dir = this_root+'CCI\\0.25\\tif\\'
        # ndvi_0_5_dir = this_root+'CCI\\0.5\\tif\\'
        mk_dir(out_dir)
        for f in os.listdir(in_dir):
            if f.endswith('.tif'):
                print(f)
                arcpy.Resample_management(in_dir+f,out_dir+f,"0.5","NEAREST")


    def arcpy_clip(self,in_raster,out_raster,in_template_dataset,nodata_value):
        # in_raster = this_root+'MRT_resample\\2000_257.txt_mosaic.hdf.FVC.tif'
        # out_raster = this_root+'test.tif'
        # in_template_dataset = this_root+'shp\\neimeng.shp'
        # nodata_value = 255
        clipping_geometry = True
        # print 'clipping'

        arcpy.Clip_management(
        in_raster=in_raster, rectangle=None, out_raster=out_raster,
            in_template_dataset=in_template_dataset, nodata_value=nodata_value,
            clipping_geometry=clipping_geometry, maintain_clipping_extent=None)


    def mapping(self,current_dir,tif,outjpeg,title,mxd_file):

        mxd = arcpy.mapping.MapDocument(mxd_file)
        df0 = arcpy.mapping.ListDataFrames(mxd)[0]

        workplace = "RASTER_WORKSPACE"

        lyr = arcpy.mapping.ListLayers(mxd, 'tif', df0)[0]
        lyr.replaceDataSource(current_dir,workplace,tif)

        for textElement in arcpy.mapping.ListLayoutElements(mxd, "TEXT_ELEMENT"):
            if textElement.name == 'title':
                textElement.text = (title)

        arcpy.mapping.ExportToJPEG(mxd,outjpeg,data_frame='PAGE_LAYOUT',df_export_width=mxd.pageSize.width,df_export_height=mxd.pageSize.height,color_mode='24-BIT_TRUE_COLOR',resolution=300,jpeg_quality=100)



class DO_func:

    def __init__(self):

        pass

    def mapping_recovery_time3(self):


        current_dir = r'D:\project05\new_2020\tif\recovery_time_in_out\\'
        tif = r'late_out_arr.tif'
        title = 'Late OUT'

        outjpeg = r'D:\project05\new_2020\jpg\in_out\{}.jpg'.format(title)
        mxd_file = r'D:\project05\MXD\recovery_time2.mxd'
        print title
        Func().mapping(current_dir,tif,outjpeg,title,mxd_file)


    def clip1(self,folder,outdir):
        # outdir = this_root+'data\\GPP\\'

        for f in os.listdir(folder):
            if f.endswith('hdf'):
                date = f.split('.')[-3]
                date = date.replace('A','')
                in_raster = folder+'\\'+f
                out_raster = outdir + date + '.tif'
                in_template_dataset = this_root+'shp\\china.shp'
                nodata_value = 65535
                Func().arcpy_clip(in_raster, out_raster, in_template_dataset, nodata_value)



    def do_clip(self,in_raster, out_raster, in_template_dataset, nodata_value):

        Func().arcpy_clip(in_raster, out_raster, in_template_dataset, nodata_value)

        pass



def main():

    args = sys.argv
    _, folder, outdir = args
    DO_func().clip1(folder,outdir)
    pass
if __name__ == '__main__':
    main()

