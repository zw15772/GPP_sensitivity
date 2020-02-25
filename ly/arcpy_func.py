# coding=gbk

import os
import arcpy
from arcpy.sa import *
import log_process
import time
import sys
from tqdm import tqdm
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types

# this_root = 'd:\\ly\\project06\\'
this_root = 'D:\\project06\\'

def mk_dir( dir, force=False):
    if not os.path.isdir(dir):
        if force == True:
            os.makedirs(dir)
        else:
            os.mkdir(dir)


class MUTIPROCESS:
    '''
    可对类内的函数进行多进程并行
    由于GIL，多线程无法跑满CPU，对于不占用CPU的计算函数可用多线程
    并行计算加入进度条
    '''

    def __init__(self, func, params):
        self.func = func
        self.params = params
        copy_reg.pickle(types.MethodType, self._pickle_method)
        pass

    def _pickle_method(self, m):
        if m.im_self is None:
            return getattr, (m.im_class, m.im_func.func_name)
        else:
            return getattr, (m.im_self, m.im_func.func_name)

    def run(self, process=-9999, process_or_thread='p', **kwargs):
        '''
        # 并行计算加进度条
        :param func: input a kenel_function
        :param params: para1,para2,para3... = params
        :param process: number of cpu
        :param thread_or_process: multi-thread or multi-process,'p' or 't'
        :param kwargs: tqdm kwargs
        :return:
        '''
        if 'text' in kwargs:
            kwargs['desc'] = kwargs['text']
            del kwargs['text']

        if process > 0:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool(process)
            elif process_or_thread == 't':
                pool = TPool(process)
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results
        else:
            if process_or_thread == 'p':
                pool = multiprocessing.Pool()
            elif process_or_thread == 't':
                pool = TPool()
            else:
                raise IOError('process_or_thread key error, input keyword such as "p" or "t"')

            results = list(tqdm(pool.imap(self.func, self.params), total=len(self.params), **kwargs))
            pool.close()
            pool.join()
            return results



class Func():

    def __init__(self):
        pass

    def resample(self,in_raster,out_raster,cell_size):
        arcpy.Resample_management(in_raster=in_raster, out_raster=out_raster, cell_size=cell_size, resampling_type="NEAREST")


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



    def kernel_clip_PDSI(self,params):
        fdir,f,outdir,in_template_dataset,nodata_value = params
        in_raster = fdir + f
        out_raster = outdir + f
        Func().arcpy_clip(in_raster, out_raster, in_template_dataset, nodata_value)
        pass

    def clip_PDSI(self):
        fdir = this_root+'NDVI\\resample\\'
        outdir = this_root+'NDVI\\clip_tif\\'
        mk_dir(outdir)
        in_template_dataset = this_root+'shp\\china.shp'
        nodata_value = -999999
        params = []
        for f in os.listdir(fdir):
            if f.endswith('tif'):
                params.append([fdir,f,outdir,in_template_dataset,nodata_value])

        MUTIPROCESS(self.kernel_clip_PDSI,params).run(process_or_thread='p')

        pass


    def kernel_do_resample(self,params):
        fdir,f,outdir,template = params
        in_raster = fdir + f
        out_raster = outdir + f
        cell_size = template
        Func().resample(in_raster, out_raster, cell_size)
        pass


    def do_resample(self):

        fdir = this_root+'NDVI\\tif\\'
        outdir = this_root+'NDVI\\resample\\'
        template = '0.05'
        mk_dir(outdir)
        params = []
        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                params.append([fdir,f,outdir,template])

        MUTIPROCESS(self.kernel_do_resample,params).run()



        pass


def main():
    # # # # # # # clip GPP LAI # # # # # # #
    # args = sys.argv
    # _, folder, outdir = args
    # DO_func().clip1(folder,outdir)
    # # # # # # # clip GPP LAI # # # # # # #

    # # # # # # # PDSI # # # # # # #
    # 1 重采样到0.05度
    # DO_func().do_resample()
    # 2 裁剪中国区域
    DO_func().clip_PDSI()
    # # # # # # # PDSI # # # # # # #

    pass

if __name__ == '__main__':
    main()

