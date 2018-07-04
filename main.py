from BatchGenerator import *
from LSTMModel import *
import matplotlib.pyplot as pl

span = 90

#domains = [['7974', 2000], ['7203', 2000], ['6752', 2000], ['6504', 2000], ['4506', 2000]]
domains = [['GBPJPY-30', 100000]]
indices = [['sma', [10]], ['sma', [30]], ['rsi', [10]], ['rsi', [30]], ['roc', [10]], ['roc', [30]],
           ['macd', [10, 30, 5]]]
#indices = [['macd', [10, 30, 5]]]
dim = (len(indices)+1)*3+1

gen = BatchGenerator(5, indices=indices, domains=domains,
                     threshould=0.03, use_fundamental=False,
                     freq=4/365, test_size=0.02, norm_distrib=True, label_mode='R')
model = LSTMModel(dim, 3, span, input_dim=dim, output_dim=1, alpha=0.00001, mode='R')

test_x, test_y = gen.getTestData(span, cut=True)
train_iter = 20001
err_list = []
acc_list = []

for i in range(train_iter):
    batch_x, batch_y = gen.getBatch(32, length=span)
    model.train(batch_x, batch_y, keep_prob=0.3)

    if i % 100 == 0:
        #error, acc = model.performance(test_x, test_y)
        error = model.performance(test_x, test_y)
        err_list.append(error)
        #acc_list.append(acc)
        #print('[iter {}] : error = {}\n\tacc = {}'.format(i, error, acc))
        print('[iter {}] : error = {}'.format(i, error))

pred = {key: model.predict(test_x[key]) for key in test_x.keys()}


for key in test_x.keys():
    pl.subplot(211)
    pl.title('Predictions of symbol : {}'.format(key))
    pl.plot(gen.datasets[key].ix[gen.splitPoint[key]:-span, ['Close']].values.flatten())
    pl.subplot(212)
    pl.plot(pred[key])
    pl.plot(test_y[key])
    pl.show()

pl.subplot(211)
pl.plot(acc_list)
pl.title('Learning curve')
pl.subplot(212)
pl.plot(err_list)
pl.show()

matrix = sum(model.confusionMatrix(test_x[key], test_y[key]) for key in test_x.keys())
print('Confusion matrix : ')
print(matrix)


command = input('Do you save the model ? [y/n] : ')

if command == 'y':
    path = input('path : ')
    print('saving the model ...')
    model.save(path)
    print('complete')
else:
    print('quit program')
