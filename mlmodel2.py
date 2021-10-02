import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('ipm.csv')

#use required features
cdf = df[['UHH','HLS','RLS','kapita','ipm']]


#Training Data and Predictor Variable
x = cdf.iloc[:, :4]
y = cdf.iloc[:, -1]
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[69.44, 14.27, 9.09 9186]]))
'''