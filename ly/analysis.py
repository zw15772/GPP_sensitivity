# coding=gbk


from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import to_raster
import ogr, os, osr
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
# import imageio
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
# import sklearn

this_root = 'D:/project06/'


class Tools:
    '''
    小工具
    '''

    def __init__(self):
        pass

    def mk_dir(self, dir, force=False):

        if not os.path.isdir(dir):
            if force == True:
                os.makedirs(dir)
            else:
                os.mkdir(dir)


    def interp_1d_1(self, val,threshold):
        # 不插离群值 只插缺失值
        if len(val) == 0 or np.std(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= threshold:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.3:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        return yi

    def pick_min_indx_from_1darray(self, arr, indexs):
        min_index = 99999
        min_val = 99999
        # plt.plot(arr)
        # plt.show()
        for i in indexs:
            val = arr[i]
            # print val
            if val < min_val:
                min_val = val
                min_index = i
        return min_index

    def index_to_mon(self, ind):

        # base_date = '198201'
        mon = ind % 12 + 1
        return mon

        pass

    def index_to_year(self,ind):

        base_year = 1982
        delta_year = ind // 12
        year = base_year + delta_year
        return year


    def growing_season_index_one_month_in_advance(self,growing_index):
        # 将生长季提前一个月
        growseason = np.array(growing_index) - 1
        new_growing_season = []
        for i in growseason:
            if i < 1:
                new_i = 12 + i
                new_growing_season.append(new_i)
            else:
                new_growing_season.append(i)
        new_growing_season = np.array(new_growing_season)
        return new_growing_season

        pass


    def cal_arrs_mean(self,arrs):

        mean_dic = {}
        for i in tqdm(range(len(arrs[0]))):
            for j in range(len(arrs[0][0])):
                key = '%04d.%04d' % (i, j)
                vals = []
                for arr in range(len(arrs)):
                    val = arrs[arr][i][j]
                    if np.isnan(val):
                        continue
                    vals.append(val)
                if len(vals) != 0:
                    mean = np.mean(vals)
                else:
                    mean = np.nan
                mean_dic[key] = mean
        mean_arr = DIC_and_TIF().pix_dic_to_spatial_arr(mean_dic)
        return mean_arr



        pass

class DIC_and_TIF:
    '''
    字典转tif
    tif转字典
    '''

    def __init__(self):

        pass


    def run(self):
        fdir = this_root+'GPP\\per_pix_anomaly\\'
        dic = {}
        plot = []
        flag = 0
        for f in os.listdir(fdir):
            if not '005' in f:
                continue
            arr = dict(np.load(fdir+f).item())
            for key in arr:
                print key,arr[key]
                if len(arr[key])>0:
                    print len(arr[key])
                    plot.append(arr[key])
                    flag += 1
                    if flag > 10:
                        break
        plot = np.array(plot)
        plot = plot.T
        plt.plot(plot)
        plt.show()
        pass

    def per_pix_dic_to_spatial_tif(self, mode, folder):

        outfolder = this_root + mode + '\\' + folder + '_tif\\'
        Tools().mk_dir(outfolder)
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        fdir = this_root + mode + '\\' + folder + '\\'
        flist = os.listdir(fdir)
        for f in flist:
            print(f)

        spatial_dic = {}
        for f in tqdm(flist):
            pix_dic = dict(np.load(fdir + f).item())
            for pix in pix_dic:
                vals = pix_dic[pix]
                spatial_dic[pix] = vals
        x = []
        y = []
        for key in spatial_dic:
            key_split = key.split('.')
            x.append(key_split[0])
            y.append(key_split[1])
        row = len(set(x))
        col = len(set(y))

        for date in tqdm(range(len(spatial_dic['0000.0000']))):
            spatial = []
            for r in range(row):
                temp = []
                for c in range(col):
                    key = '%04d.%04d' % (r, c)
                    val_pix = spatial_dic[key][date]
                    temp.append(val_pix)
                spatial.append(temp)
            spatial = np.array(spatial)
            grid = np.isnan(spatial)
            grid = np.logical_not(grid)
            spatial[np.logical_not(grid)] = -999999
            to_raster.array2raster(outfolder + '%03d.tif' % date, originX, originY, pixelWidth, pixelHeight, spatial)
            # plt.imshow(spatial)
            # plt.colorbar()
            # plt.show()

        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        # spatial = []
        # all_vals = []
        # for r in tqdm(range(row)):
        #     temp = []
        #     for c in range(col):
        #         key = '%04d.%04d' % (r, c)
        #         val_pix = spatial_dic[key]
        #         temp.append(val_pix)
        #         all_vals.append(val_pix)
        #     spatial.append(temp)

        pass

    def arr_to_tif(self, array, newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = -999999
        to_raster.array2raster(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass

    def arr_to_tif_GDT_Byte(self, array, newRasterfn):
        # template
        tif_template = this_root + 'conf\\tif_template.tif'
        _, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        grid_nan = np.isnan(array)
        grid = np.logical_not(grid_nan)
        array[np.logical_not(grid)] = 255
        to_raster.array2raster_GDT_Byte(newRasterfn, originX, originY, pixelWidth, pixelHeight, array)
        pass


    def spatial_arr_to_dic(self,arr):

        pix_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                pix = '%04d.%04d'%(i,j)
                val = arr[i][j]
                pix_dic[pix] = val

        return pix_dic


    def pix_dic_to_spatial_arr(self, spatial_dic):

        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        tif_template = this_root + 'conf\\tif_template.tif'
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        row = len(arr_template)
        col = len(arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = '%04d.%04d' % (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        # hist = []
        # for v in all_vals:
        #     if not np.isnan(v):
        #         if 00<v<1.5:
        #             hist.append(v)

        spatial = np.array(spatial,dtype=float)
        return spatial
        # # plt.figure()
        # # plt.hist(hist,bins=100)
        # # plt.title(str(set_level))
        # plt.figure()
        # # spatial = np.ma.masked_where(spatial<0,spatial)
        # # spatial = np.ma.masked_where(spatial>2,spatial)
        # # plt.imshow(spatial,'RdBu_r',vmin=0.7 ,vmax=1.3)
        # plt.imshow(spatial, 'RdBu_r')
        # plt.colorbar()
        # # plt.title(str(set_level))
        # plt.show()


    def pix_dic_to_spatial_arr_ascii(self, spatial_dic):
        # dtype can be in ascii format
        # x = []
        # y = []
        # for key in spatial_dic:
        #     key_split = key.split('.')
        #     x.append(key_split[0])
        #     y.append(key_split[1])
        # row = len(set(x))
        # col = len(set(y))
        tif_template = this_root + 'conf\\tif_template.tif'
        arr_template, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        row = len(arr_template)
        col = len(arr_template[0])
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = '%04d.%04d' % (r, c)
                if key in spatial_dic:
                    val_pix = spatial_dic[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        spatial = np.array(spatial)
        return spatial


    def pix_dic_to_tif(self, spatial_dic, out_tif):

        spatial = self.pix_dic_to_spatial_arr(spatial_dic)
        # spatial = np.array(spatial)
        self.arr_to_tif(spatial, out_tif)

    def spatial_tif_to_lon_lat_dic(self):
        tif_template = this_root + 'conf\\SPEI.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        # print(originX, originY, pixelWidth, pixelHeight)
        # exit()
        pix_to_lon_lat_dic = {}
        for i in tqdm(range(len(arr))):
            for j in range(len(arr[0])):
                pix = '%04d.%04d' % (i, j)
                lon = originX + pixelWidth * j
                lat = originY + pixelHeight * i
                pix_to_lon_lat_dic[pix] = [lon, lat]
        print('saving')
        np.save(this_root + 'arr\\pix_to_lon_lat_dic', pix_to_lon_lat_dic)

    def void_spatial_dic(self):
        tif_template = this_root + 'conf\\tif_template.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        void_dic = {}
        for row in range(len(arr)):
            for col in range(len(arr[row])):
                key = '%04d.%04d' % (row, col)
                void_dic[key] = []
        return void_dic

    def ascii_to_arr(self,lonlist,latlist,vals):
        '''
        transform ascii text to spatial array
        :param lonlist:[.....]
        :param latlist: [.....]
        :param vals: [.....]
        :return:
        '''

        lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        lon_lat_dic_reverse = {}
        for key in lon_lat_dic:
            lon,lat = lon_lat_dic[key]
            new_key = str(lon)+'_'+str(lat)
            lon_lat_dic_reverse[new_key] = key

        spatial_dic = {}
        for i in range(len(lonlist)):
            lt = str(lonlist[i])+'_'+str(latlist[i])
            pix = lon_lat_dic_reverse[lt]
            spatial_dic[pix] = vals[i]

        arr = self.pix_dic_to_spatial_arr_ascii(spatial_dic)
        return arr


    def plot_back_ground_arr(self):
        tif_template = this_root + 'conf\\tif_template.tif'
        arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif_template)
        back_ground = []
        for i in range(len(arr)):
            temp = []
            for j in range(len(arr[0])):
                val = arr[i][j]
                if val > 60000:
                    temp.append(np.nan)
                else:
                    temp.append(1)
            back_ground.append(temp)
        back_ground = np.array(back_ground)
        plt.imshow(back_ground, 'gray', vmin=0, vmax=1.4)

        # return back_ground

        pass


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




class Winter:
    '''
    主要思想：
    1、计算每个NDVI像素每个月的多年平均值
    2、计算值大于3000的月的个数
    3、如果大于3000的个数大于10，则没有冬季，反之则有冬季
    4、选出冬季date range
    '''
    def __init__(self):

        pass


    def run(self):
        # self.cal_monthly_mean()
        # self.count_num()
        # self.get_grow_season_index()
        # self.composite_tropical_growingseason()
        # self.check_composite_growing_season()
        # self.check_pix()
        self.growing_season_one_month_in_advance()
        pass

    def cal_monthly_mean(self):

        outdir = this_root+'NDVI\\mon_mean_tif\\'
        Tools().mk_dir(outdir)
        fdir = this_root+'NDVI\\clip_tif\\'
        for m in tqdm(range(1,13)):
            arrs_sum = 0.
            for y in range(1982,2016):
                date = '{}{}'.format(y,'%02d'%m)
                tif = fdir+date+'.tif'
                arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
                arr[arr<-9999] = np.nan
                arrs_sum += arr
            mean_arr = arrs_sum/len(range(1982,2016))
            mean_arr = np.array(mean_arr,dtype=float)
            DIC_and_TIF().arr_to_tif(mean_arr,outdir+'%02d.tif'%m)

    def count_num(self):
        # 计算tropical区域
        fdir = this_root + 'NDVI\\mon_mean_tif\\'
        # pix_lon_lat_dic = dict(np.load(this_root + 'arr\\pix_to_lon_lat_dic.npy').item())
        arrs = []
        month = range(1,13)
        for m in tqdm(month):
            tif = fdir+'%02d.tif'%m
            arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(tif)
            arrs.append(arr)

        row = len(arrs[0])
        col = len(arrs[0][0])

        winter_count = []
        winter_pix = []
        for i in tqdm(range(row)):
            temp = []
            for j in range(col):
                flag = 0
                for arr in arrs:
                    val = arr[i][j]
                    if val>5000:
                        flag += 1.
                if flag == 12:
                    winter_pix.append('%04d.%04d'%(i,j))
                temp.append(flag)
            winter_count.append(temp)

        np.save(this_root+'NDVI\\tropical_pix',winter_pix)


        ##### show #####
        winter_count = np.array(winter_count)
        winter_count = np.ma.masked_where(winter_count<12,winter_count)
        plt.imshow(winter_count,'jet')
        plt.colorbar()
        plt.show()
        pass


    def max_5_vals(self,vals):

        vals = np.array(vals)
        # 从小到大排序，获取索引值
        a = np.argsort(vals)
        maxvs = []
        maxv_ind = []
        for i in a[-5:][::-1]:
            maxvs.append(vals[i])
            maxv_ind.append(i)
        # 南半球
        if 0 in maxv_ind or 1 in maxv_ind:
            if 9 in maxv_ind:
                growing_season = [0, 8, 9, 10, 11]
            elif 10 in maxv_ind:
                growing_season = [0, 1, 2, 10, 11]
            elif 11 in maxv_ind:
                growing_season = [0, 1, 2, 3, 11]
            else:
                mid = int(np.mean(maxv_ind))
                growing_season = [mid-2,mid-1,mid,mid+1,mid+2]
        # 北半球
        else:
            mid = int(np.mean(maxv_ind))
            growing_season = [mid-2,mid-1,mid,mid+1,mid+2]
        growing_season = np.array(growing_season) + 1
        return growing_season

    def get_grow_season_index(self):
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        fdir = this_root + 'NDVI\\mon_mean_tif\\'
        arrs = []
        month = range(1, 13)
        for m in month:
            tif = fdir + '%02d.tif' % m
            arr, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(tif)
            arrs.append(arr)

        row = len(arrs[0])
        col = len(arrs[0][0])

        winter_dic = {}
        for i in tqdm(range(row)):
            for j in range(col):
                # if i < 150:
                #     continue
                pix = '%04d.%04d' % (i, j)
                if pix in tropical_pix:
                   continue
                vals = []
                for arr in arrs:
                    val = arr[i][j]
                    vals.append(val)
                if vals[0] > -10000:
                    std = np.std(vals)
                    if std == 0:
                        continue
                    growing_season = self.max_5_vals(vals)
                    # print growing_season
                    # plt.plot(vals)
                    # plt.grid()
                    # plt.show()
                    winter_dic[pix] = growing_season
        np.save(this_root+'NDVI\\growing_season_index',winter_dic)

        pass

    def composite_tropical_growingseason(self):
        growing_season_index = dict(np.load(this_root + 'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root + 'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for i in growing_season_index:
            pix_dic[i] = growing_season_index[i]
        for pix in tropical_pix:
            # pix_dic[pix] = 2
            pix_dic[pix] = range(1,13)
        np.save(this_root+'NDVI\\composite_growing_season',pix_dic)

    def growing_season_one_month_in_advance(self):
        # 将生长季提前一个月

        growing_season_f = this_root + 'NDVI\\composite_growing_season.npy'
        growing_season_dic = dict(np.load(growing_season_f).item())
        new_growing_season_dic = {}
        for pix in tqdm(growing_season_dic):
            growing_season = growing_season_dic[pix]
            new_growing_season = Tools().growing_season_index_one_month_in_advance(growing_season)
            new_growing_season_dic[pix] = new_growing_season
        np.save(this_root + 'NDVI\\composite_growing_season_one_month_in_advance.npy',new_growing_season_dic)
        pass

    def check_pix(self):
        growing_season_index = dict(np.load(this_root+'NDVI\\growing_season_index.npy').item())
        tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        pix_dic = {}
        for pix in tropical_pix:
            pix_dic[pix] = 2

        for pix in growing_season_index:
            pix_dic[pix] = 1
            # print growing_season_index[pix]

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(pix_dic)

        plt.imshow(arr)
        plt.show()

        pass


    def check_composite_growing_season(self):

        f = this_root+'NDVI\\composite_growing_season.npy'
        dic = dict(np.load(f).item())
        spatial_dic = {}

        for pix in dic:
            val = len(dic[pix])
            spatial_dic[pix] = val

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr)
        plt.show()




class NDVI:

    def __init__(self):

        pass

    def run(self):
        tif = this_root+r'tif\recovery_time\pick_post_growing_season_events_plot_gen_recovery_time\global.tif'
        out_tif = this_root+r'tif\recovery_time\pick_post_growing_season_events_plot_gen_recovery_time\global_mask.tif'
        # self.gen_max_NDVI_arr()
        # self.gen_NDVI_mask_pix()
        self.check_NDVI_mask_pix()

    def gen_max_NDVI_arr(self):
        outf = this_root+'NDVI\\max_NDVI'
        fdir = this_root+'NDVI\\mon_mean_tif\\'
        arrs = []
        for f in os.listdir(fdir):
            arr,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir+f)
            arrs.append(arr)

        spatial_arr = []
        for i in tqdm(range(len(arrs[0]))):
            temp = []
            for j in range(len(arrs[0][0])):
                one_pix = []
                for k in range(len(arrs)):
                    val = arrs[k][i][j]
                    if val > -999:
                        one_pix.append(val)
                if len(one_pix) != len(arrs):
                    max_val = np.nan
                else:
                    max_val = np.max(one_pix)
                temp.append(max_val)
            spatial_arr.append(temp)
        spatial_arr = np.array(spatial_arr)
        # spatial_arr[spatial_arr<2000]=np.nan
        # plt.imshow(spatial_arr)
        # plt.colorbar()
        # plt.show()
        np.save(outf,spatial_arr)


    def gen_NDVI_mask_pix(self):
        f = this_root+'NDVI\\max_NDVI.npy'
        outf = this_root+'NDVI\\NDVI_mask_pix'
        arr = np.load(f)
        arr[arr<2000] = np.nan

        dic = DIC_and_TIF().spatial_arr_to_dic(arr)
        new_dic = {}
        for key in dic:
            val = dic[key]
            if np.isnan(val):
                continue
            new_dic[key] = val
        np.save(outf,new_dic)

        pass


    def check_NDVI_mask_pix(self):
        f = this_root+'NDVI\\NDVI_mask_pix.npy'
        dic = dict(np.load(f).item())

        arr = DIC_and_TIF().pix_dic_to_spatial_arr(dic)
        plt.imshow(arr)
        plt.show()





class SMOOTH:
    '''
    一些平滑算法
    '''

    def __init__(self):

        pass

    def interp_1d(self, val):
        if len(val) == 0:
            return [None]

        # 1、插缺失值
        x = []
        val_new = []
        flag = 0
        for i in range(len(val)):
            if val[i] >= -10:
                flag += 1.
                index = i
                x = np.append(x, index)
                val_new = np.append(val_new, val[i])
        if flag / len(val) < 0.9:
            return [None]
        interp = interpolate.interp1d(x, val_new, kind='nearest', fill_value="extrapolate")

        xi = range(len(val))
        yi = interp(xi)

        # 2、利用三倍sigma，去除离群值
        # print(len(yi))
        val_mean = np.mean(yi)
        sigma = np.std(yi)
        n = 3
        yi[(val_mean - n * sigma) > yi] = -999999
        yi[(val_mean + n * sigma) < yi] = 999999
        bottom = val_mean - n * sigma
        top = val_mean + n * sigma
        # plt.scatter(range(len(yi)),yi)
        # print(len(yi),123)
        # plt.scatter(range(len(yi)),yi)
        # plt.plot(yi)
        # plt.show()
        # print(len(yi))

        # 3、插离群值
        xii = []
        val_new_ii = []

        for i in range(len(yi)):
            if -999999 < yi[i] < 999999:
                index = i
                xii = np.append(xii, index)
                val_new_ii = np.append(val_new_ii, yi[i])

        interp_1 = interpolate.interp1d(xii, val_new_ii, kind='nearest', fill_value="extrapolate")

        xiii = range(len(val))
        yiii = interp_1(xiii)

        # for i in range(len(yi)):
        #     if yi[i] == -999999:
        #         val_new_ii = np.append(val_new_ii, bottom)
        #     elif yi[i] == 999999:
        #         val_new_ii = np.append(val_new_ii, top)
        #     else:
        #         val_new_ii = np.append(val_new_ii, yi[i])

        return yiii

    def smooth_convolve(self, x, window_len=11, window='hanning'):
        """
        1d卷积滤波
        smooth the data using a window with requested size.
        This method is based on the convolution of a scaled window with the signal.
        The signal is prepared by introducing reflected copies of the signal
        (with the window size) in both ends so that transient parts are minimized
        in the beginning and end part of the output signal.
        input:
            x: the input signal
            window_len: the dimension of the smoothing window; should be an odd integer
            window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
        output:
            the smoothed signal
        example:
        t=linspace(-2,2,0.1)
        x=sin(t)+randn(len(t))*0.1
        y=smooth(x)
        see also:
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        scipy.signal.lfilter
        NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        """
        x = np.array(x)

        if x.ndim != 1:
            raise ValueError, "smooth only accepts 1 dimension arrays."

        if x.size < window_len:
            raise ValueError, "Input vector needs to be bigger than window size."

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        # return y
        return y[(window_len / 2 - 1):-(window_len / 2)]

    def smooth(self, x):
        # 后窗滤波
        # 滑动平均
        x = np.array(x)
        temp = 0
        new_x = []
        for i in range(len(x)):
            if i + 3 == len(x):
                break
            temp += x[i] + x[i + 1] + x[i + 2] + x[i + 3]
            new_x.append(temp / 4.)
            temp = 0
        return np.array(new_x)

    def forward_window_smooth(self, x, window=3):
        # 前窗滤波
        # window = window-1
        # 不改变数据长度

        if window < 0:
            raise IOError('window must be greater than 0')
        elif window == 0:
            return x
        else:
            pass

        x = np.array(x)

        new_x = np.array([])
        # plt.plot(x)
        # plt.show()
        for i in range(len(x)):
            if i - window < 0:
                new_x = np.append(new_x, x[i])
            else:
                temp = 0
                for w in range(window):
                    temp += x[i - w]
                smoothed = temp / float(window)
                new_x = np.append(new_x, smoothed)
        return new_x

    def filter_3_sigma(self, arr_list):
        sum_ = []
        for i in arr_list:
            if i >= 0:
                sum_.append(i)
        sum_ = np.array(sum_)
        val_mean = np.mean(sum_)
        sigma = np.std(sum_)
        n = 3
        sum_[(val_mean - n * sigma) > sum_] = -999999
        sum_[(val_mean + n * sigma) < sum_] = -999999

        # for i in
        return sum_

        pass



class PICK_events:
    def __init__(self):
        pass

    def run(self):
        # self.pick()
        # self.check_events()
        self.check_events_is_in_growing_season()
        pass

    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        key = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            # if val < -0.5:# SPEI
            if val < -1:  # PDSI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 3:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []
        # print(len(pdsi))
        # print(event_i)
        if len(event_i) > 3:
            events.append(event_i)

        flag = 0
        events_dic = {}

        # 取两个端点
        for i in events:
            # print(i)
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                # print(jj)
                if jj - 1 >= 0:
                    new_i.append(jj - 1)
                else:
                    pass
            new_i.append(i[-1])
            if i[-1] + 1 < len(pdsi):
                new_i.append(i[-1] + 1)
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()

            min_val = min(vals)
            if -2 <= min_val < -1:
                level = 1
            elif -3 <= min_val < -2:
                level = 2
            elif -4 <= min_val < -3:
                level = 3
            elif min_val <= -4.:
                level = 4
            else:
                print('error')
                print(vals)
                print(min_val)
                time.sleep(1)
                continue
            min_index = Tools().pick_min_indx_from_1darray(pdsi,new_i)
            # print level
            # print new_i
            # print min_val
            # print min_index
            # plt.plot(pdsi)
            # plt.grid()
            # plt.show()
            # exit()
            events_dic[flag] = [level, new_i, min_index]
        return events_dic, key


    def pick(self):
        spei_dir = this_root + 'PDSI\\per_pix_smooth\\'
        out_dir = this_root + 'PDSI\\events\\'
        Tools().mk_dir(out_dir, force=True)
        for f in tqdm(os.listdir(spei_dir), 'file...'):
            # if not '005' in f:
            #     continue
            spei_dic = dict(np.load(spei_dir + f).item())
            single_event_dic = {}
            for pix in spei_dic:
                spei = spei_dic[pix]

                if spei[0] < -999:
                    single_event_dic[pix] = []
                    continue

                # spei = Tools().forward_window_smooth(spei, 3)
                params = [spei, pix]
                events_dic, key = self.kernel_find_drought_period(params)
                # for i in events_dic:
                #     print i,events_dic[i]
                # exit()
                events = []  # 严重干旱事件
                for i in events_dic:
                    level, date_range, min_index = events_dic[i]
                    events.append({'level':level, 'date_range':date_range, 'min_index':min_index})

                # for eventi in events:
                #     print eventi
                # plt.plot(spei)
                # plt.grid()
                # plt.show()
                # exit()

                # # # # # # # # # # # # # # # # # # # # # # #
                # 不筛选单次事件（前后n个月无干旱事件）
                single_event_dic[pix] = events
                # print events_4
                # # # # # # # # # # # # # # # # # # # # # # #

                # # # # # # # # # # # # # # # # # # # # # # #
                # # 筛选单次事件（前后n个月无干旱事件）
                # single_event = []
                # for i in range(len(events_4)):
                #     if i - 1 < 0:  # 首次事件
                #         if events_4[i][0] - n < 0 or events_4[i][-1] + n >= len(spei):  # 触及两边则忽略
                #             continue
                #         if len(events_4) == 1:
                #             single_event.append(events_4[i])
                #         elif events_4[i][-1] + n <= events_4[i + 1][0]:
                #             single_event.append(events_4[i])
                #         continue
                #
                #     # 最后一次事件
                #     if i + 1 >= len(events_4):
                #         if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= len(spei):
                #             single_event.append(events_4[i])
                #         break
                #
                #     # 中间事件
                #     if events_4[i][0] - events_4[i - 1][-1] >= n and events_4[i][-1] + n <= events_4[i + 1][0]:
                #         single_event.append(events_4[i])
                # single_event_dic[pix] = single_event
                # # # # # # # # # # # # # # # # # # # # # # #
            np.save(out_dir + f, single_event_dic)



    def check_events(self):
        fdir = this_root+'PDSI\\events\\'
        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())
            for key in dic:
                events = dic[key]
                if len(events) != 0:
                    print events


    def check_events_is_in_growing_season(self):
        # 计算在生长季内发生干旱的事件数

        growing_season_index_f = this_root+'NDVI\\composite_growing_season.npy'
        growing_season_index_dic = dict(np.load(growing_season_index_f).item())

        outf = this_root+'PDSI\\growing_season_drought_events_counts'

        events_dir = this_root+'PDSI\\events\\'


        count_dic = {}
        for f in tqdm(os.listdir(events_dir)):
            dic = dict(np.load(events_dir+f).item())
            for pix in dic:
                if not pix in growing_season_index_dic:
                    continue
                growing_season = growing_season_index_dic[pix]
                events = dic[pix]
                if len(events) == 0:
                    continue
                flag = 0
                for event in events:
                    min_index = event['min_index']
                    current_mon = Tools().index_to_mon(min_index)
                    if current_mon in growing_season:
                        flag += 1.
                count_dic[pix] = flag
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(count_dic)
        np.save(outf,arr)
        arr[arr==0] = np.nan

        DIC_and_TIF().plot_back_ground_arr()
        plt.figure(figsize=(14,10))
        plt.imshow(arr,'jet')
        plt.colorbar()
        # plt.show()
        plt.savefig(this_root+'png\\check_events_is_in_growing_season.png',ppi=600)











        pass

class Sensitivity:

    def __init__(self):
        self.NDVI_mask_pix = dict(np.load(this_root+'NDVI\\NDVI_mask_pix.npy').item())
        # self.tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        # self.tropical_pix = np.load(this_root+'NDVI\\tropical_pix.npy')
        # plt.imshow(self.tropical_pix)
        # plt.show()
        pass

    def run(self):
        # self.check_sensitivity()
        # self.sensitivity()
        # self.sensitivity_annual()
        # self.get_non_drought_GPP_mean()
        # self.png_sensitivity_annual()
        self.png_sensitivity_5_year()
        pass


    def check_sensitivity(self):
        f = this_root+'arr\\gpp_non_drought_mean.npy'
        arr = np.load(f)
        arr[arr<-9999] = np.nan
        cmap = sns.diverging_palette(236, 0, s=99, l=50, n=10, center="light")
        cmap = mpl.colors.ListedColormap(cmap)
        # plt.imshow(arr,cmap=cmap,vmin=0.7,vmax=1.3)
        plt.imshow(arr,cmap)
        plt.colorbar()
        plt.show()
        pass

    def sensitivity(self):
        pdsi_dir = this_root+'PDSI\\per_pix_smooth\\'
        event_dir = this_root+'PDSI\\events\\'
        veg_dir = this_root+'data\\GPP\\per_pix_smooth\\'
        # veg_dir = this_root+'data\\GPP\\per_pix_anomaly_smooth\\'
        spatial_dic = {}
        for f in tqdm(os.listdir(event_dir)):
            # if not '050' in f:
            #     continue
            event_dic = dict(np.load(event_dir+f).item())
            veg_dic = dict(np.load(veg_dir+f).item())
            # pdsi_dic = dict(np.load(pdsi_dir+f).item())
            for pix in event_dic:
                events = event_dic[pix]
                if len(events) == 0:
                    continue
                if not pix in self.NDVI_mask_pix:
                    continue
                veg_val = veg_dic[pix]
                events = event_dic[pix]
                # pdsi_val = pdsi_dic[key]
                drought_veg_vals = []
                non_drought_veg_vals = []
                all_event_i = []
                for event in events:
                    date_range = event['date_range']
                    min_index = event['min_index']
                    for i in date_range:
                        all_event_i.append(i)
                # exit()
                for i in range(len(veg_val)):
                    if i in all_event_i:
                        drought_veg_vals.append(veg_val[i])
                    else:
                        non_drought_veg_vals.append(veg_val[i])
                # print drought_veg_vals
                # print non_drought_veg_vals
                drought_veg_vals_mean = np.mean(drought_veg_vals)
                non_drought_veg_vals_mean = np.mean(non_drought_veg_vals)
                ratio = drought_veg_vals_mean/non_drought_veg_vals_mean
                spatial_dic[pix] = ratio
                # print ratio
                # print key
                # spatio_dic[]
                # plt.plot(pdsi_val,c='r')
                # plt.twinx()
                # plt.plot(veg_val,c='green')
                # plt.grid()
                # plt.show()
        # spatio_dic
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        np.save(this_root+'arr\\sensitivity_all_year',arr)
        plt.imshow(arr,'jet')
        plt.colorbar()
        plt.show()
                # exit()


    def get_non_drought_GPP_mean(self):
        # 计算在非干旱时期GPP的平均值，一张tif图
        # 条件1：gpp val 不在干旱index中
        # 条件2：gpp val 不在非生长季
        # 计算GPP 的平均值
        growing_season_f = this_root + 'NDVI\\composite_growing_season.npy'
        growing_season_dic = dict(np.load(growing_season_f).item())
        event_dir = this_root + 'PDSI\\events\\'
        veg_dir = this_root + 'data\\GPP\\per_pix_smooth\\'

        outf = this_root+'arr\\gpp_non_drought_mean'
        gpp_mean_spatial = {}
        for f in tqdm(os.listdir(event_dir)):
            # if not '005' in f:
            #     continue
            event_dic = dict(np.load(event_dir + f).item())
            veg_dic = dict(np.load(veg_dir + f).item())
            # pdsi_dic = dict(np.load(pdsi_dir+f).item())
            for pix in event_dic:
                events = event_dic[pix]
                if len(events) == 0:
                    continue
                if not pix in self.NDVI_mask_pix:
                    continue
                veg_val = veg_dic[pix]
                events = event_dic[pix]
                growing_season = growing_season_dic[pix]
                # 将生长季提前一个月
                growing_season = Tools().growing_season_index_one_month_in_advance(growing_season)
                # pdsi_val = pdsi_dic[key]
                drought_veg_vals = []
                non_drought_indx = []

                all_event_i = [] # 干旱事件的index月
                for event in events:
                    date_range = event['date_range']
                    min_index = event['min_index']
                    # 筛选在生长季内发生的干旱事件
                    current_month = Tools().index_to_mon(min_index)
                    if not current_month in growing_season:
                        continue
                    for dr in date_range:
                        all_event_i.append(dr)
                # print all_event_i

                selected_val = []
                for i in range(len(veg_val)):
                    # 条件1：gpp val 不在干旱index中
                    if i in all_event_i:
                        # selected_val.append(np.nan)
                        continue
                    this_month = Tools().index_to_mon(i)
                    # 条件2：gpp val 不在非生长季
                    if not this_month in growing_season:
                        # selected_val.append(np.nan)
                        continue
                    selected_val.append(veg_val[i])
                gpp_mean = np.mean(selected_val)
                gpp_mean_spatial[pix] = gpp_mean
                # print growing_season
                # print selected_val
                # plt.plot(selected_val,alpha=0.5,linewidth=4)
                # plt.plot(veg_val,alpha=0.5)
                # plt.show()
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(gpp_mean_spatial)
        np.save(outf,arr)
        plt.imshow(arr)
        plt.colorbar()
        plt.show()

        pass


    def sensitivity_annual(self):
        outf = this_root+'arr\\sensitivity_annual'
        growing_season_f = this_root+'NDVI\\composite_growing_season_one_month_in_advance.npy'
        growing_season_dic = dict(np.load(growing_season_f).item())

        ##### plot composite_growing_season_one_month_in_advance #####
        # growing_season_dic_count = {}
        # for key in growing_season_dic:
        #     val = len(growing_season_dic[key])
        #     growing_season_dic_count[key] = val
        #
        # DIC_and_TIF().plot_back_ground_arr()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(growing_season_dic_count)
        # plt.imshow(arr)
        #
        # plt.colorbar()
        # plt.show()
        ##### plot composite_growing_season_one_month_in_advance #####


        non_drought_gpp_mean_f = this_root+'arr\\gpp_non_drought_mean.npy'
        non_drought_gpp_mean_arr = np.load(non_drought_gpp_mean_f)
        non_drought_gpp_mean_arr[non_drought_gpp_mean_arr<-9999] = np.nan
        non_drought_gpp_mean_dic = DIC_and_TIF().spatial_arr_to_dic(non_drought_gpp_mean_arr)
        event_dir = this_root + 'PDSI\\events\\'
        veg_dir = this_root + 'data\\GPP\\per_pix_smooth\\'
        # veg_dir = this_root+'data\\GPP\\per_pix_anomaly_smooth\\'
        ratio_dic = {}
        for f in tqdm(os.listdir(event_dir)):
            # if not '050' in f:
            #     continue
            event_dic = dict(np.load(event_dir + f).item())
            veg_dic = dict(np.load(veg_dir + f).item())
            # pdsi_dic = dict(np.load(pdsi_dir+f).item())
            pix_year_event_dic = {}
            '''
            生长季的干旱事件
            pix_year_event_dic format:
            key = 0705.0724 
            val = {1984: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33], 
                   1986: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52], 
                   1987: [63, 64, 65, 66, 67, 68]}
            '''

            for pix in event_dic:
                events = event_dic[pix]
                if len(events) == 0:
                    continue
                if not pix in self.NDVI_mask_pix:
                    continue
                year_event_dic = {}
                # growing_season = growing_season_dic[pix]
                for event in events:
                    min_index = event['min_index']
                    # 筛选在生长季内发生的干旱事件
                    current_month = Tools().index_to_mon(min_index)
                    # if not current_month in growing_season:
                    #     continue
                    current_year = Tools().index_to_year(min_index)
                    year_event_dic[current_year]=event['date_range']
                # for i in year_event_dic:
                #     print pix,i,year_event_dic[i]
                pix_year_event_dic[pix] = year_event_dic


            for pix in pix_year_event_dic:
                year_dic = pix_year_event_dic[pix]
                growing_season = growing_season_dic[pix]
                veg_val = veg_dic[pix]
                non_drought_gpp_mean = non_drought_gpp_mean_dic[pix]
                if np.isnan(non_drought_gpp_mean):
                    continue
                # print pix,year_dic
                for year in year_dic:
                    drought_range = year_dic[year]
                    selected_veg_val = []
                    for dr in drought_range:
                        mon = Tools().index_to_mon(dr)
                        if mon in growing_season:
                            # print year,dr,mon,growing_season
                            selected_veg_val.append(veg_val[dr])
                    selected_veg_val_mean = np.mean(selected_veg_val)
                    ratio = selected_veg_val_mean/non_drought_gpp_mean
                    # print pix,year,ratio
                    key = pix+'_'+str(year)
                    ratio_dic[key] = ratio
        print '\nsaving...'
        np.save(outf,ratio_dic)


    def png_sensitivity_annual(self):
        f = this_root+'arr\\sensitivity_annual.npy'
        out_png_dir = this_root+'png\\sensitivity_annual\\'
        Tools().mk_dir(out_png_dir,force=True)
        annual_ratio_dic = dict(np.load(f).item())
        void_spatio_dic = DIC_and_TIF().void_spatial_dic()
        years = range(1982,2018)
        for year in tqdm(years):
            plt.figure(figsize=(14,8))
            spatio_dic = {}
            for pix in void_spatio_dic:
                key = pix+'_'+str(year)
                if key in annual_ratio_dic:
                    val = annual_ratio_dic[key]
                    spatio_dic[pix] = val
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatio_dic)
            arr[arr>2] = np.nan
            DIC_and_TIF().plot_back_ground_arr()
            plt.imshow(arr,'jet',vmin=0.8,vmax=1.2)
            plt.colorbar()
            plt.title(str(year)+' ratio')
            # plt.show()
            plt.savefig(out_png_dir+str(year)+'_ratio.png',ppi=144)
            plt.close()



    def png_sensitivity_5_year(self):
        f = this_root+'arr\\sensitivity_annual.npy'
        out_png_dir = this_root+'png\\sensitivity_5_year\\'
        Tools().mk_dir(out_png_dir,force=True)
        annual_ratio_dic = dict(np.load(f).item())
        void_spatio_dic = DIC_and_TIF().void_spatial_dic()
        years = range(1982,2018)
        years_list = []
        for i in range(len(years)/5):
            temp = years[i*5:(i+1)*5]
            years_list.append(temp)
        years_list[-1].append(2017)

        for yl in years_list:
            plt.figure(figsize=(14, 8))
            DIC_and_TIF().plot_back_ground_arr()
            arrs = []
            for year in yl:
                spatio_dic = {}
                for pix in void_spatio_dic:
                    key = pix+'_'+str(year)
                    if key in annual_ratio_dic:
                        val = annual_ratio_dic[key]
                        spatio_dic[pix] = val
                arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatio_dic)
                arr[arr>2] = np.nan
                arrs.append(arr)
            mean_arr = Tools().cal_arrs_mean(arrs)
            plt.imshow(mean_arr, 'jet', vmin=0.8, vmax=1.2)
            plt.colorbar()
            plt.title(str(yl[0]) + '-' + str(yl[-1])+' ratio')
            # plt.show()
            plt.savefig(out_png_dir + str(yl[0]) + '-' + str(yl[-1]) + '_ratio.png', ppi=600)
            plt.close()

def main():
    # Winter().run()
    # PICK_events().run()
    Sensitivity().run()
    # NDVI().run()

    pass


if __name__ == '__main__':

    main()