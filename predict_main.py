from LSTMModel import *
from BatchGenerator import *
import matplotlib.pyplot as pl

def Gain(pred, values):
    actions = np.argmax(pred, axis=1)
    order = [0,0]
    gain = 0
    history = []
    for i in range(len(actions)):
        a = 1 if actions[i] == 2 else (-1 if actions[i] == 0 else 0)
        if a and order[0] != a:
            g = order[0]*(values[i] - order[1])
            gain += g
            print('[time {}] : change possession. pos={}, gain={}'.format(i, a, g))
            order = [a, values[i]]
        history.append(gain)
    
    print('Total gain = {}'.format(gain))
    pl.subplot(311)
    pl.plot(values)
    pl.subplot(312)
    pl.plot(actions)
    pl.subplot(313)
    pl.plot(history)
    pl.show()
    

model = LSTMModel(5, 3, 90, input_dim=5, output_dim=3, alpha=0.00005)

path = input('path to the model : ')

print('loading model ...')
model.restore(path)
print('complete')

gen = BatchGenerator(30, indices=[['sma', [10]], ['sma', [30]], ['rsi', [10]], ['roc', [10]]],
                     test_size=0.05,
                     domains=[['GBPJPY-30', 100000]],#[['7974', 2000], ['7203', 2000], ['6752', 2000], ['6504', 2000], ['4506', 2000]],#
                     threshould=0.001, use_fundamental=False,
                     freq=4/365)#1/120)
x, y = gen.get_test_data(90, cut=False)

while True:
    symbol = input('The symbol you want to predict : ')
    if symbol == 'end':
        break

    pred = model.predict(x[symbol][-90:])
    pl.plot(pred)
    pl.show()
    pred = model.predict(x[symbol])
    Gain(pred, gen.get_price(symbol)[gen.splitPoint[symbol]:])
