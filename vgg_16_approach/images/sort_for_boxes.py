import pandas as pd

cars = pd.read_csv('beforeYOLO_MaxOnly_train+test.csv')
#print(cars)

cars['SX'] = cars.CX - (cars.W)/2
cars['SY'] = cars.CY - (cars.H)/2
cars['EX'] = cars.CX +  (cars.W)/2
cars['EY'] = cars.CY + (cars.H)/2

#cars.drop([CX,CY,w,h])
cars = cars[['IMAGE', 'SX', 'SY', 'EX', 'EY', 'CLASS']]

#print(cars)
cars.to_csv('train_test_vgg_format.csv',index=False)

