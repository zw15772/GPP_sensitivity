# coding=gbk
'''
author: LiYang
Date: 20200222
Location: zzq BeiJing
'''
from scipy import interpolate
from scipy import signal
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import os
from analysis import *

this_root = 'd:\\project06\\'


class CLIP():

    def __init__(self):

        pass


    def kernel_clip_GLASS(self,pamams):
        years_dir, i, arcpy, script = pamams
        folder = years_dir + i + '\\'
        cmd = [arcpy, script, folder]
        cmd = ' '.join(cmd)
        os.system(cmd)

        pass

    def clip_GLASS(self):
        arcpy = r'C:\Python27_for_arcgis\ArcGIS10.2\python.exe'
        script = r'C:\Users\ly\PycharmProjects\GPP_sensitivity\ly\arcpy_func.py'

        years_dir = this_root + 'origin_data\\GPP\\'
        outdir = this_root + 'data\\GPP\\'
        Tools().mk_dir(outdir, force=True)
        pamams = []
        for i in os.listdir(years_dir):
            pamams.append([years_dir, i, arcpy, script])
        MUTIPROCESS(self.kernel_clip_GLASS,pamams).run()
def main():
    CLIP().clip_GLASS()

    pass


if __name__ == '__main__':
    main()





