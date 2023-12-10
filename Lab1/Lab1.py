import pandas as pd
import numpy as np

titanic = pd.read_csv('C:/Users/Phoenix/Desktop/Titanic.csv', delimiter=',', index_col='PassengerId')
print(titanic['Fare'].describe())
