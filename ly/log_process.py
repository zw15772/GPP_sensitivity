# coding=gbk

import logging
from logging import handlers
import sys
import math
import time
import datetime
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import copy_reg
import types



class MUTIPROCESS:
    '''
    �ɶ����ڵĺ������ж���̲���
    ����GIL�����߳��޷�����CPU�����ڲ�ռ��CPU�ļ��㺯�����ö��߳�
    ���м�����������
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
        # ���м���ӽ�����
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




class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#��־�����ϵӳ��

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#������־��ʽ
        self.logger.setLevel(self.level_relations.get(level))#������־����
        sh = logging.StreamHandler()#����Ļ�����
        sh.setFormatter(format_str) #������Ļ����ʾ�ĸ�ʽ
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#���ļ���д��#ָ�����ʱ���Զ������ļ��Ĵ�����
        #ʵ����TimedRotatingFileHandler
        #interval��ʱ������backupCount�Ǳ����ļ��ĸ����������������������ͻ��Զ�ɾ����when�Ǽ����ʱ�䵥λ����λ�����¼��֣�
        # S ��
        # M ��
        # H Сʱ��
        # D �졢
        # W ÿ���ڣ�interval==0ʱ��������һ��
        # midnight ÿ���賿
        th.setFormatter(format_str)#�����ļ���д��ĸ�ʽ
        self.logger.addHandler(sh) #�Ѷ���ӵ�logger��
        self.logger.addHandler(th)



def process_bar(i,length,time_init=None,start_time=None,end_time=None,custom_print=''):
    '''

    :param i: ��ǰѭ�� i int
    :param length: ��ѭ������ int
    :param time_init: ��ʼʱ�� time
    :param start_time: ÿ��ѭ����ʼʱ�� time
    :param end_time: ÿ��ѭ������ʱ�� time
    :param custom_print: �Զ���print str
    :return:
    '''

    i = i + 1
    if time_init:
        time_delta = end_time - start_time
        time_elapse = end_time - time_init

        eta = time_elapse / (float(i) / length)


        done = int(50 * (i) / length)
        sys.stdout.write(
            '\r%s '%changeTime(time_elapse)+ #��ȥʱ��
            "[%s%s]" % ('=' * done + '>'+'%0.2f' % (100 * float(i) / length) + '%', '<'+'-' * (50 - done)) + #������+�ٷֱ�
            ' eta %s'%changeTime(eta)+#ʣ��ʱ��
            '\t' +str(custom_print)
        )
        sys.stdout.flush()
    else:
        done = int(50 * (i) / length)
        sys.stdout.write(
            "\r[%s%s]" % ('=' * done + '>'+'%0.2f' % (100 * float(i) / length) + '%', '<'+'-' * (50 - done))+  # ������+�ٷֱ�
            '\t'+str(custom_print)
        )
        sys.stdout.flush()


def process_bar_1(iterable):
    iterable = range(50)


    pass



def changeTime(allTime):
    # print(allTime)
    day = 24*60*60
    hour = 60*60
    min = 60
    if allTime <60:
        return "%ds"%math.ceil(allTime)
    elif allTime > day:
        days = divmod(allTime,day)
        return "%dd %s"%(int(days[0]),changeTime(days[1]))
    elif allTime > hour:
        hours = divmod(allTime,hour)
        return '%dh %s'%(int(hours[0]),changeTime(hours[1]))
    else:
        mins = divmod(allTime,min)
        return "%dm %ds" % (int(mins[0]), math.ceil(mins[1]))

def main():
    P = process_bar
    time_init = time.time()
    for i in range(1000):
        start = time.time()
        time.sleep(0.01)
        end = time.time()
        P(i, 1000,time_init,start,end,i)
        # start = time.time()


    # a = changeTime(100.46546)
    # print(a)

if __name__ == '__main__':
    main()
    # for i in range(10):
    #     add_time(i)
    #     time.sleep(1)
