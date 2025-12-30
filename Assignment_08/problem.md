your objective is to write an LSTM algorithm that predicts the stock price some number of days in the future. Extract a stock that has a stock symbol that has the same first letter as your last (of if you don’t have a last name your first name). Your objective is to pull data for the stock from yfinance and build train and test a model that predicts the stock price in the future as many days as possible. Use at least a year of data. Find the train and test accuracy, and number of days in advance you can predict the stock value accurately. To determine this, you want to make a plot of prediction (test) accuracy versus number of days. Choose a reasonable minimum accuracy and report this number of days. Submit train and test accuracy, and number of days in advance you can predict, and the plot.

Hint: I used this code to set the lookback and delta to the price being predicted, so if lookback is 30 and delta is 1 you are using the previous 30 days to predict tomorrows price, if delta is 5 you would be using the previous 30 days to predict the price 5 days in the future
from sklearn.model_selection import train_test_split
def split_data_1o(data_raw, lookback, Delta ):
data_raw = data_raw.numpy() # convert to numpy array
data = []
# create all possible sequences of length seq_len
for index in range(len(data_raw) - lookback):
data.append(data_raw[index: index + lookback]) # as separate lists in list data
xdata = np.array(data); # turn this list into an array
N = len(xdata[:,0])
y = xdata[1:N,0,Delta-1].reshape(-1,1) ## choose future value y
## randomly select test and train sets
x_train, x_test, y_train, y_test = train_test_split(xdata[0:N-1],y,test_size=0.3,random_state=0)
return [x_train, y_train, x_test, y_test]