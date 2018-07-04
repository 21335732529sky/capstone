import pandas as pd
import numpy as np
import tools
from tqdm import tqdm
import random
import matplotlib.pyplot as pl
import sys
from collections import Counter

class BatchGenerator:
    def __init__(self, span, indices=None, domains=None, test_size=0.2,
                 threshould=0.01, use_fundamental=False, freq=4/365,
                 norm_distrib=False, label_mode='C'):
        self.span = span
        self.th = threshould
        self.norm = norm_distrib
        tmp = self.readStockPrices(domains)

        self.prices = {key: tmp[key].ix[30:, ['Close']].values.flatten() for key in tmp.keys()}
        stocks = self.maxScaling(self.readStockPrices(domains))
        stocks = {key: self.calcIndices(df, indices) for key,df in zip(stocks.keys(),
                                                                       stocks.values())}

        if use_fundamental:
            seats = self.maxScaling(self.readBalanceSeat([x[0] for x in domains]))
            self.datasets = self.mergeDataFrames(stocks, seats, list(stocks.keys()))
        else:
            self.datasets = stocks

        self.splitPoint = {key: int((1 - test_size)*length) for key,length in domains}
        self.index = {key: list(range(l)) for key,l in zip(self.splitPoint.keys(),
                                                           self.splitPoint.values())}
        self.smooth = {key: self.LowPassFilter(self.datasets[key].ix[:, ['Close']].values.flatten(),
                                               freq, self.datasets[key].shape[0]+300)\
                       for key in self.datasets.keys()}
        self.labels = {key: self.label(self.smooth[key], span, mode=label_mode) for key in self.datasets.keys()}

        '''
        pl.subplot(211)
        pl.plot(self.smooth['7974'][-200:])
        pl.subplot(212)
        pl.plot(self.labels['7974'][-200:])
        pl.show()
        '''


        for key in self.labels.keys() :
            #self.datasets[key].ix[:, ['Close']] = np.array([[x] for x in self.smooth[key][:-300]])
            self.datasets[key] = self.datasets[key][30:]
            self.labels[key] = self.labels[key][30:-300]

        if label_mode == 'C': self.clustered = self.cluster(self.labels, min_=91)

        self.mode = label_mode

    def Func(self, x):
        if x == 'sma':
            return tools.calc_sma
        elif x == 'wma':
            return tools.calc_wma
        elif x == 'rsi':
            return tools.calc_rsi
        elif x == 'macd':
            return tools.calc_macd
        elif x == 'roc':
            return tools.calc_roc

    def cluster(self, labels, min_=0):
        ret = {}
        for key in labels.keys():
            ret[key] = [
                [i for i,l in enumerate(np.argmax(labels[key][:self.splitPoint[key]], axis=1)) if l == 0 and i > min_],
                [i for i,l in enumerate(np.argmax(labels[key][:self.splitPoint[key]], axis=1)) if l == 1 and i > min_],
                [i for i,l in enumerate(np.argmax(labels[key][:self.splitPoint[key]], axis=1)) if l == 2 and i > min_]
            ]

        return ret

    def calcIndices(self, df, indices):
        ret = pd.DataFrame(df.ix[:, :])
        for name, params in indices:
            tmp = []
            print('calculating {} ...'.format(name))
            f = self.Func(name)
            for i in tqdm(range(df.shape[0]), total=df.shape[0]):
                st = max(0, i-200)
                ed = i+1
                tmp2 = f(df.ix[st:ed, ['Close']].values.flatten(),
                                       *params)
                if type(tmp2) != tuple: tmp2 = (tmp2, )
                tmp2 = [x[-1] for x in tmp2]
                tmp.append(tmp2)
            tmp = np.array(tmp).T
            for i, data in enumerate(tmp): ret[name+str(params[0])+'_{}'.format(i)] = data
            for i, data in enumerate(tmp): ret[name+str(params[0])+'^2_{}'.format(i)] = np.array(list(map(lambda x : x**2, data)))
            for i, data in enumerate(tmp): ret[name+str(params[0])+'^3_{}'.format(i)] = np.array(list(map(lambda x : x**2, data)))
            #ret[name+str(params[0])+"^2"] = np.array(list(map(lambda x : x**2, tmp)))
            #ret[name+str(params[0])+"^3"] = np.array(list(map(lambda x : x**3, tmp)))

        return ret



    def LowPassFilter(self, values, th, length):
        fft = np.fft.fft(list(values)+[values[-1],]*(length-values.shape[0]), n=length)
        mask = [1 if i/fft.shape[0] <= th or i/fft.shape[0] >= 1 - th else 0\
                for i in range(fft.shape[0])]
        fft = np.array([f*m for f,m in zip(fft, mask)])
        return np.array(abs(np.fft.ifft(fft)))

    def readStockPrices(self, domains):
        ret = {}
        for symbol, length in domains:
            df = pd.read_csv('CSVData/{}.csv'.format(symbol), usecols=['Date', 'Close'],
                             index_col='Date')
            df = df.ix[-length:, :]
            ret[symbol] = df

        return ret

    def readBalanceSeat(self, symbols):
        ret = {}
        for symbol in symbols:
            df = pd.read_csv('balance_seat/{}.csv'.format(symbol), index_col='Date')
            ret[symbol] = df

        return ret

    def mergeDataFrames(self, df1, df2, keys):
        ret = {}
        for key in keys:
            df1[key].index = pd.to_datetime(df1[key].index).ceil('D')
            df2[key].index = pd.to_datetime(df2[key].index).ceil('D')
            tmp = pd.merge(df1[key], df2[key], how='left', left_index=True, right_index=True)
            tmp = tmp.fillna(method='ffill')
            tmp = tmp.fillna(method='bfill')
            ret[key] = tmp
            print(ret[key])

        return ret

    def maxScaling(self, data):
        ret = {}
        for key in data.keys():
            ret[key] = data[key] / data[key].max()

        return ret

    def getBatch(self, batch_size, length=0):
        ret = []
        ret_l = []
        indexes = self.getIndex(batch_size, norm_distrib=self.norm, length=length)
        for symbol, time in indexes:
            ret.append(list(self.datasets[symbol].ix[time-length:time, :].values))
            ret_l.append(self.labels[symbol][time])

        if self.mode == 'R':
            ret_l = [[x] for x in ret_l]


        return np.array(ret), np.array(ret_l)

    def getIndex(self, batch_size, norm_distrib=False, length=0):
        if self.mode == 'C':
            if norm_distrib:
                symbol = [random.choice(list(self.clustered.keys())) for _ in range(batch_size)]
                sel_label = [random.choice([0,1,2]) for _ in range(batch_size)]
                ret = [[s, random.choice(self.clustered[s][l])] for s,l in zip(symbol, sel_label)]
            else:
                symbol = [random.choice(list(self.clustered.keys())) for _ in range(batch_size)]
                ret = [[s, random.choice(np.array(self.clustered[s]).flatten()[0])] for s in symbol]
        else:
            symbol = [random.choice(list(self.labels.keys())) for _ in range(batch_size)]
            ret = [[s, random.choice(np.array(range(length, self.splitPoint[s] - length)))] for s in symbol]

        return ret


    def getTestData(self, length, cut=True):
        ret_x = {}
        ret_y = {}
        for key in self.datasets.keys():
            tmp = []
            for i in range(self.splitPoint[key], self.datasets[key].shape[0] -cut*length):
                tmp.append(self.datasets[key].ix[i-length:i].values)
            ret_x[key] = np.array(tmp)
            ret_y[key] = np.array(self.labels[key][self.splitPoint[key]:-cut*length])
            if self.mode == 'R': ret_y[key] = np.array([[x] for x in ret_y[key]])
        return ret_x, ret_y

    def getPrice(self, key):
        return self.prices[key]

    def label(self, values, span, mode='C'):
        ret = []

        if mode == 'C':
            band = int(span/2)
            for i in range(band, values.shape[0]-band):
                left = 1 if values[i-band]/values[i] < 1 - self.th else \
                       (-1 if values[i-band]/values[i] > 1 + self.th else 0)
                right = -1 if values[i+band]/values[i] < 1 - self.th else \
                       (1 if values[i+band]/values[i] > 1 + self.th else 0)
                if left + right >= 1 : ret.append([0,0,1])
                elif left + right == 0 : ret.append([0,1,0])
                else : ret.append([1,0,0])

            return np.array([ret[0],]*band + ret + [ret[-1],]*band)


        elif mode == 'R':
            ret = [(values[i+span]/values[i] - 1) for i in range(values.shape[0] - span)]

            return np.array(ret + [ret[-1],]*span)




        #return np.array(ret+[ret[-1],]*span)
