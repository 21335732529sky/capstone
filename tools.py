import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def read_data(mode, base='USDJPY.csv'):
    filename = os.listdir('Datasets_{}/'.format(mode))

    try: filename.remove(base)
    except ValueError:
        print('Error: base data is not exist')
        return None

    df = pd.read_csv('Datasets_{}/{}'.format(mode, base),
                     usecols=['Date','Close'], index_col='Date')
    df.columns = [base.replace('.csv', '')]

    for n in []:
        tmp = pd.read_csv('Datasets_{}/{}'.format(mode, n),
                         usecols=['Date', 'Close'], index_col='Date')
        tmp.columns = [n.replace('.csv', '')]
        df = df.join(tmp)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')

    return df

def normalize(df, method='std'):
    tmp = df.values
    tcol = df.columns
    tind = df.index
    if method == 'std':
        tmp = np.array([StandardScaler().fit_transform(x) for x in tmp.T]).T
        tdf = pd.DataFrame(tmp, columns=tcol, index=tind)
    if method == 'ret':
        df = df.ix[1:,:]/df.ix[:-1,:].values
        tdf = df
    return tdf

def accuracy(y_test, pred):
    return sum(1 for i in range(pred.shape[0]) if y_test[i] == pred[i])/pred.shape[0]

def log_loss(label, logits):
    loss = 0
    for y,l in zip(logits, label):
        loss += -y[np.argmax(l)]*np.log(y[np.argmax(l)])

    return loss / len(label)

#----------------------------------------------------------------------
#                       technical indicator
#----------------------------------------------------------------------
def calc_sma(Close, window):
    if Close.shape[0] < window: return Close
    sma = pd.DataFrame(Close).rolling(center=False, window=window).mean()
    sma = sma.fillna(method='bfill').values
    return sma.flatten()

def calc_wma(Close, window, norm=False):
    ret = []
    for i in range(Close.shape[0]):
        if i < window:
            ret.append(sum((j+1)*v for j,v in enumerate(Close[:i])) / sum(range(1, i + 2)))
        else:
            ret.append(sum((j+1)*v for j,v in enumerate(Close[i - window:i])) / sum(range(1, window + 1)))

    return np.array(ret) if not norm else np.array(ret) / Close

def calc_macd(Close, span_s, span_l, span_sig):
    V = pd.DataFrame(Close)
    short = pd.DataFrame(pd.ewma(V, span=span_s)).fillna(method='bfill')
    Long = pd.DataFrame(pd.ewma(V, span=span_l)).fillna(method='bfill')
    macd = short-Long
    macd_sig = pd.DataFrame(pd.ewma(macd, span=span_sig)).fillna(method='bfill')

    return macd.values.flatten(), macd_sig.values.flatten()

def calc_dmi(High, Low, Close, window):
    ret = []
    pdi = []
    mdi = []
    for j in range(len(High) - window):
        pDM = sum(High[i] - High[i - 1] if High[i] > High[i - 1]\
                  else 0 for i in range(j+1, window+j+1))
        mDM = sum(Low[i - 1] - Low[i] if Low[i - 1] > Low[i]\
                  else 0 for i in range(j+1, window+j+1))
        TR = sum(max(High[i] - Low[i],
                     High[i] - Close[i - 1],
                     Close[i - 1] - Low[i])\
                for i in range(j+1, window+j+1))
        pDI = pDM / TR * 100
        mDI = mDM / TR * 100
        pdi.append(pDI)
        mdi.append(mDI)
        ret.append((pDI-mDI)/(pDI+mDI))
    return np.array([ret[0] for _ in range(window)] + ret),\
           np.array([pdi[0] for _ in range(window)] + pdi),\
           np.array([mdi[0] for _ in range(window)] + mdi)

def calc_roc(Close, span):
    if Close.shape[0] <= span:
        return Close
    ret = [v / u - 1 for v,u in zip(Close[span:], Close[:-span])]
    return np.array([ret[0] for _ in range(span)] + ret)

def calc_stochastic(High, Low, Close,
                    span_x, span_y, span_z):
    CLn=[Close[i] - min(Low[i - span_x:i]) for i in range(span_x, len(High))]
    CLn = [CLn[0],] * span_x + CLn
    HnLn=[max(High[i - span_x:i]) - min(Low[i - span_x:i])
          for i in range(span_x, len(High))]
    HnLn = [HnLn[0],] * span_x + HnLn
    CLn=pd.DataFrame(CLn).fillna(method='bfill')
    HnLn=pd.DataFrame(HnLn).fillna(method='bfill')
    per_k=(CLn/HnLn).fillna(method='bfill')
    per_d=(CLn.rolling(span_y).mean()/HnLn.rolling(span_y).mean()).fillna(method='bfill')
    s_per_d=per_d.rolling(span_z).mean().fillna(method='bfill')

    return per_k.values.flatten(), per_d.values.flatten(), s_per_d.values.flatten()

def calc_slope(Close, span):

    sumX = sum(range(span))
    sumX2 = sum([x**2 for x in range(span)])
    sumY = np.array([sum(Close[i - span:i]) for i in range(span, Close.shape[0] + 1)])
    sumXY = np.array([sum(Close[i - span + j] * j for j in range(span)) \
                      for i in range(span, Close.shape[0] + 1)])
    ret = [(span*Sxy - sumX*Sy)/(span*sumX2 - sumX**2)\
           for Sy, Sxy in zip(sumY, sumXY)]

    return np.array([ret[0],]*(span-1) + ret)

def calc_rsi(Close, span):
    if Close.shape[0] < span:
        return Close
    Close = np.array([0] + list(Close[1:] - Close[:-1]))
    sumabs = [sum(abs(x) for x in Close[i - span:i]) for i in range(span, Close.shape[0] + 1)]
    sumup = [sum(x for x in Close[i - span:i] if x > 0) for i in range(span, Close.shape[0] + 1)]

    tmp = [a/b for a,b in zip(sumup, sumabs)]

    return np.array([tmp[0],]*(span-1) + tmp)

def calc_bollinger(Close, span, y):
    ret = []
    for i in range(span, Close.shape[0] + 1):
        mean = sum(Close[i - span:i]) / span
        std = (sum((v - mean) ** 2 for v in Close[i - span:i]) / span) ** 0.5
        ret.append(mean + std*y)
    return np.array([ret[0],]*(span-1) + ret)

def calc_cci(High, Low, Close, span):
    ret = []
    pt = [(h + l + c) / 3 for h,l,c in zip(High, Low, Close)]
    for i in range(span, High.shape[0] + 1):
        psma = sum(pt[i-span:i]) / span
        l1 = sum(abs(p - psma) for p in pt[i-span:i])
        ret.append(1/0.015 * (pt[i-1]-psma)/l1)

    return np.array([ret[0],]*(span-1) + ret)

def calc_williams(Close, span):
    tmp =[(Close[i - 1] - max(Close[i - span:i])) / (max(Close[i - span:i]) - min(Close[i - span:i])) \
          for i in range(span, Close.shape[0] + 1)]
    return np.array([tmp[0],]*(span-1) + tmp)

def calc_ichimoku(High, Low, p_x, p_y, p_z):
    turning = [(max(High[i - p_x:i]) + min(Low[i - p_x:i])) / 2 for i in range(p_x, High.shape[0])]
    base = [(max(High[i - p_y:i]) + min(Low[i - p_y:i])) / 2 for i in range(p_y, High.shape[0])]

    span1 = [(t + b)/2 for t, b in zip(turning[p_y-p_x:], base)]
    span1 = [0,]*(p_y)*2 + span1

    span2 = [(max(High[i - p_z:i]) + min(Low[i - p_z:i])) / 2 for i in range(p_z, High.shape[0])]

    span2 = [0,]*(p_y + p_z) + span2
    turning = [0,]*p_x + turning
    base = [0,]*p_y + base

    return np.array(turning), np.array(base), np.array(span1[:-p_y]), np.array(span2[:-p_y])
