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

# this_root = 'd:\\ly\\project06\\'
this_root = 'D:/project06/'


class CLIP():

    def __init__(self):

        pass


    def kernel_clip_GLASS(self,pamams):
        years_dir, i, arcpy, script, outdir = pamams
        folder = years_dir + i + '\\'
        cmd = [arcpy, script, folder, outdir]
        cmd = ' '.join(cmd)
        os.system(cmd)

        pass

    def clip_GLASS(self):
        arcpy = r'C:\Python27_for_arcgis\ArcGIS10.2\python.exe'
        script = r'D:\ly\GPP_sensitivity\ly\arcpy_func.py'

        years_dir = this_root + 'origin_data\\LAI\\'
        outdir = this_root + 'data\\LAI\\'
        Tools().mk_dir(outdir, force=True)
        pamams = []
        for i in os.listdir(years_dir):
            pamams.append([years_dir, i, arcpy, script, outdir])
        MUTIPROCESS(self.kernel_clip_GLASS,pamams).run()



    def clip_PDSI(self):
        arcpy = r'C:\Python27_for_arcgis\ArcGIS10.2\python.exe'
        script = r'D:\ly\GPP_sensitivity\ly\arcpy_func.py'



        pass



class Pre_Process:

    def __init__(self):
        pass

    def run(self):
        # fdir = this_root+'GLASS\\data\\monthly\\LAI\\'
        # fdir = this_root+'PDSI\\clip_tif\\'
        # per_pix = this_root+'data\\LAI\\per_pix\\'
        # per_pix = this_root+'PDSI\\per_pix\\'
        anomaly = this_root+'data\\GPP\\per_pix\\'
        # anomaly = this_root+'PDSI\\per_pix\\'
        # anomaly_smooth = this_root+'data\\GPP\\per_pix_anomaly_smooth\\'
        # anomaly_smooth = this_root+'PDSI\\per_pix_smooth\\'
        anomaly_smooth = this_root+'data\\GPP\\per_pix_smooth\\'
        # # Tools().mk_dir(outdir)
        # self.data_transform(fdir,per_pix)
        # self.cal_anomaly(per_pix,anomaly)
        # self.check_ndvi_anomaly()
        # self.check_per_pix(anomaly_smooth)
        self.smooth_anomaly(anomaly,anomaly_smooth)

        pass


    def day_8_to_monthly(self):
        '''
        时间分辨率8天转换到月
        :return:
        '''

        fdir = this_root+'GLASS\\data\\8days\\LAI\\'
        outdir = this_root+'GLASS\\data\\monthly\\LAI\\'
        Tools().mk_dir(outdir,force=1)
        date_dic = {}
        for y in range(1982,2018):
            for m in range(1,13):
                key = '%s%02d'%(y,m)
                date_dic[key] = []

        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                year_base = int(f.split('.')[0][:4])
                time_base = datetime.datetime(year_base,1,1)
                time_delta = int(f.split('.')[0][4:]) - 1
                this_day = time_base+datetime.timedelta(time_delta)
                year = this_day.year
                month = this_day.month
                key = '%s%02d'%(year,month)
                if key in date_dic:
                    date_dic[key].append(f)

        for date in tqdm(date_dic):
            # print date
            # print outdir+date+'.tif'
            this_month = date_dic[date]
            arr = 0.
            flag = 0.
            for tif in this_month:
                array,originX,originY,pixelWidth,pixelHeight = to_raster.raster2array(fdir+tif)
                # print originX,originY,pixelWidth,pixelHeight
                array = np.array(array,dtype=float)
                array[array>60000] = np.nan
                arr += array
                flag += 1
            if flag == 0:
                continue
            mean_arr = arr / flag
            DIC_and_TIF().arr_to_tif(mean_arr,outdir+date+'.tif')

        pass


    def do_data_transform(self):
        father_dir = this_root + 'SPEI\\tif\\'
        for spei_dir in os.listdir(father_dir):
            print spei_dir + '\n'
            interval = spei_dir[-2:]

            spei_dir_ = spei_dir.upper()[:4] + '_' + interval
            outdir = this_root + 'SPEI\\per_pix\\' + spei_dir_ + '\\'
            print outdir
            Tools().mk_dir(outdir)
            self.data_transform(father_dir + spei_dir + '\\', outdir)

    def data_transform(self, fdir, outdir):
        # 不可并行，内存不足
        Tools().mk_dir(outdir)
        # 将空间图转换为数组
        # per_pix_data
        flist = os.listdir(fdir)
        date_list = []
        for f in flist:
            if f.endswith('.tif'):
                date = f.split('.')[0]
                date_list.append(date)
        date_list.sort()
        all_array = []
        for d in tqdm(date_list, '1/3 loading...'):
            # for d in date_list:
            for f in flist:
                if f.endswith('.tif'):
                    if f.split('.')[0] == d:
                        # print(d)
                        array, originX, originY, pixelWidth, pixelHeight = to_raster.raster2array(fdir + f)
                        all_array.append(array)

        row = len(all_array[0])
        col = len(all_array[0][0])

        void_dic = {}
        void_dic_list = []
        for r in range(row):
            for c in range(col):
                void_dic['%04d.%04d' % (r, c)] = []
                void_dic_list.append('%04d.%04d' % (r, c))

        # print(len(void_dic))
        # exit()
        for r in tqdm(range(row), '2/3 transforming...'):
            for c in range(col):
                for arr in all_array:
                    val = arr[r][c]
                    void_dic['%04d.%04d' % (r, c)].append(val)

        # for i in void_dic_list:
        #     print(i)
        # exit()
        flag = 0
        temp_dic = {}
        for key in tqdm(void_dic_list, '3/3 saving...'):
            flag += 1
            # print('saving ',flag,'/',len(void_dic)/100000)
            temp_dic[key] = void_dic[key]
            if flag % 10000 == 0:
                # print('\nsaving %02d' % (flag / 10000)+'\n')
                np.save(outdir + 'per_pix_dic_%03d' % (flag / 10000), temp_dic)
                temp_dic = {}
        np.save(outdir + 'per_pix_dic_%03d' % 0, temp_dic)

    def kernel_cal_anomaly(self, params):
        fdir, f, save_dir = params
        pix_dic = dict(np.load(fdir + f).item())
        anomaly_pix_dic = {}
        for pix in pix_dic:
            ####### one pix #######
            vals = pix_dic[pix]
            # 清洗数据
            vals = Tools().interp_1d_1(vals,0)

            if len(vals) == 1:
                anomaly_pix_dic[pix] = []
                continue
            climatology_means = []
            climatology_std = []
            # vals = signal.detrend(vals)
            for m in range(1, 13):
                one_mon = []
                for i in range(len(pix_dic[pix])):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(pix_dic[pix][i])
                mean = np.mean(one_mon)
                std = np.std(one_mon)
                climatology_means.append(mean)
                climatology_std.append(std)

            # 算法1
            # pix_anomaly = {}
            # for m in range(1, 13):
            #     for i in range(len(pix_dic[pix])):
            #         mon = i % 12 + 1
            #         if mon == m:
            #             this_mon_mean_val = climatology_means[mon - 1]
            #             this_mon_std_val = climatology_std[mon - 1]
            #             if this_mon_std_val == 0:
            #                 anomaly = -999999
            #             else:
            #                 anomaly = (pix_dic[pix][i] - this_mon_mean_val) / float(this_mon_std_val)
            #             key_anomaly = i
            #             pix_anomaly[key_anomaly] = anomaly
            # arr = pandas.Series(pix_anomaly)
            # anomaly_list = arr.to_list()
            # anomaly_pix_dic[pix] = anomaly_list

            # 算法2
            pix_anomaly = []
            for i in range(len(vals)):
                mon = i % 12
                std_ = climatology_std[mon]
                mean_ = climatology_means[mon]
                if std_ == 0:
                    anomaly = 0 ##### 修改gpp
                else:
                    anomaly = (vals[i] - mean_) / std_

                pix_anomaly.append(anomaly)
            # pix_anomaly = Tools().interp_1d_1(pix_anomaly,-100)
            # plt.plot(pix_anomaly)
            # plt.show()
            anomaly_pix_dic[pix] = pix_anomaly

        np.save(save_dir + f, anomaly_pix_dic)

    def cal_anomaly(self,fdir,save_dir):
        # fdir = this_root + 'NDVI\\per_pix\\'
        # save_dir = this_root + 'NDVI\\per_pix_anomaly\\'
        Tools().mk_dir(save_dir)
        flist = os.listdir(fdir)
        # flag = 0
        params = []
        for f in flist:
            # print(f)
            params.append([fdir, f, save_dir])

        # for p in params:
        #     print(p[1])
        #     self.kernel_cal_anomaly(p)
        MUTIPROCESS(self.kernel_cal_anomaly, params).run(process=6, process_or_thread='p',
                                                         text='calculating anomaly...')

    def kernel_smooth_anomaly(self,params):
        fdir,f,outdir = params
        dic = dict(np.load(fdir + f).item())
        smooth_dic = {}
        for key in dic:
            vals = dic[key]
            smooth_vals = SMOOTH().forward_window_smooth(vals)
            smooth_dic[key] = smooth_vals
        np.save(outdir + f, smooth_dic)

    def smooth_anomaly(self,fdir,outdir):
        # fdir = this_root+'data\\GPP\\per_pix_anomaly\\'
        # outdir = this_root+'data\\GPP\\per_pix_anomaly_smooth\\'
        Tools().mk_dir(outdir)
        params = []
        for f in tqdm(os.listdir(fdir)):
            params.append([fdir,f,outdir])
        MUTIPROCESS(self.kernel_smooth_anomaly,params).run()



    def check_ndvi_anomaly(self):
        fdir = this_root + 'NDVI\\per_pix\\'
        for f in os.listdir(fdir):
            dic = dict(np.load(fdir+f).item())

            for pix in tqdm(dic):
                val = dic[pix]
                std = np.std(val)
                if std == 0 or len(val) == 0:
                    continue
                # print val
                val = Tools().interp_1d_1(val,-3000)
                # print val
                if len(val) == 1:
                    continue
                plt.plot(val)
                plt.grid()
                plt.show()
        pass

    def check_per_pix(self,fdir):

        for f in os.listdir(fdir):
            print f
            dic = dict(np.load(fdir+f).item())
            for pix in dic:
                val = dic[pix]
                if len(val) == 0:
                    continue
                if val[0]<0:
                    continue
                print pix,val
                plt.plot(val)
                plt.show()

    def extend_GPP(self):
        fidr = this_root + 'GPP\\per_pix_anomaly\\'
        outdir = this_root + 'GPP\\per_pix_anomaly_extend\\'
        Tools().mk_dir(outdir)
        for f in tqdm(os.listdir(fidr)):
            # if not '015' in f:
            #     continue
            dic = dict(np.load(fidr + f).item())
            new_dic = {}
            for key in dic:
                val = dic[key]
                n = len(val)
                if n == 0:
                    new_dic[key] = []
                    continue
                ni = 408 - 192
                null_list = [np.nan] * ni
                null_list.extend(val)
                new_dic[key] = null_list
            np.save(outdir + f, new_dic)







def main():
    # CLIP().clip_GLASS()
    Pre_Process().run()
    pass


if __name__ == '__main__':
    main()





