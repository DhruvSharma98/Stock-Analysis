import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib import style
style.use("ggplot")

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

#user defined
import cursor_functionality


###############################################################################

''' Function Declarations '''

def insert_row(date, df, dummy_df):
    df_up = df[df.index > date]
    df_low = df[df.index < date]
    df_up = df_up.append(dummy_df.loc[date])
    df_result = pd.concat([df_up,df_low])
    return df_result


#for macd
def cal_factor(n):
    factor = 2/(n+1)
    return factor

def moving_avg(df, n, column, col_name, start):
    fac = cal_factor(n)
    avg = (df.loc[start:start+(n-1), column].sum())/n
    df[col_name] = pd.Series(index=df.index)
    df.at[start+(n-1), col_name] = avg
    df.fillna(0, inplace=True)
    for count,ele in enumerate(df.loc[(start+n):df.shape[0], col_name],
                               start+n):
        df.loc[count, col_name] = (((df.loc[count, column]
                                    - df.loc[count-1, col_name])*fac)
                                    + df.loc[count-1, col_name])


#for rsi
def cal_movement(df, n, column, col_name):
    fac = n-1
    avg = (df.loc[0:(n-1), column].sum())/n
    df[col_name] = pd.Series(index=df.index)
    df.at[n-1, col_name] = avg
    df.fillna(0, inplace=True)
    for count,ele in enumerate(df.loc[n:df.shape[0], col_name], n):
        df.loc[count, col_name] = (((df.loc[count-1, col_name]*fac)
                                   +df.loc[count, column]) / n)

#for cci
def mean_deviation(df, period):
    df['ADV'] = pd.Series(index=df.index)
    for i in range(period-1, df.shape[0]):
        df.loc[i, 'ADV'] = (sum(abs((df.loc[i:i-(period-1):-1, 'Typical_Price']
                                    - df.loc[i, 'SMA']))) / period)

###############################################################################

''' Creating Dataframe '''

df = pd.read_excel("YesBankDataset.xlsx", index_col=0)
pd.options.display.max_columns = None
pd.set_option('display.min_rows', 20)
pd.set_option('display.max_rows', None)
##print(df)
##print(df.describe())
##print(df.dtypes)

''' Inserting Rows '''

dummy_s = pd.date_range(start='2005-06-30', end='2019-11-22', freq='B')
null_array = np.empty((dummy_s.size,df.shape[1]))
null_array[:] = np.nan
data = null_array

dummy_df = pd.DataFrame(data, index=dummy_s, columns=df.columns)
dummy_df.index.name = 'Date'
dummy_df.sort_index(axis='index', ascending=False, inplace=True)

for date in dummy_df.index:
    if date not in df.index:
        df = insert_row(date, df, dummy_df)

df.interpolate(method='linear', inplace=True)
df = df.resample('MS').mean()


###############################################################################

''' Calculating MACD '''

df1 = pd.DataFrame.copy(df, deep=True)
df1.reset_index(inplace=True)

n_fast = 12
n_slow = 26
n_signal = 9

moving_avg(df1, n_fast, 'Close', 'EMA_'+str(n_fast), start=0)
moving_avg(df1, n_slow, 'Close', 'EMA_'+str(n_slow), start=0)
df1['MACD'] = (df1.loc[n_slow-1:, 'EMA_'+str(n_fast)]
               - df1.loc[n_slow-1:, 'EMA_'+str(n_slow)])

moving_avg(df1, n_signal, 'MACD', 'Signal_Line', start=n_slow-1)

df1['Hist'] = (df1.loc[n_slow+n_signal-2:,'MACD']
               - df1.loc[n_slow+n_signal-2:, 'Signal_Line'])

df1.fillna(0, inplace=True)
df1.set_index('Date', inplace=True)


###############################################################################

''' Calculating RSI '''

df2 = pd.DataFrame.copy(df, deep=True)
df2.reset_index(inplace=True)

df2['Change'] = df2['Close'].diff(periods=1)
df2['Upward_Movement'] = pd.Series(index=df.index)
df2['Downward_Movement'] = pd.Series(index=df.index)


for count, i in enumerate(df2['Change']):
    if i>0:
        df2.loc[count, 'Upward_Movement'] = i
    elif i<=0:
        df2.loc[count, 'Upward_Movement'] = 0
        
    if i<0:
        df2.loc[count, 'Downward_Movement'] = abs(i)
    elif i>=0:
        df2.loc[count, 'Downward_Movement'] = 0

cal_movement(df2, 14, 'Upward_Movement', 'Avg_Upward_Movement')
cal_movement(df2, 14, 'Downward_Movement', 'Avg_Downward_Movement')
df2.fillna(0, inplace=True)
df2['Relative_Strength'] = (df2['Avg_Upward_Movement']
                            / df2['Avg_Downward_Movement'])
df2['RSI'] = 100-(100/(df2['Relative_Strength']+1))
df2.fillna(0, inplace=True)

moving_avg(df2, 30, 'Close', 'EMA_Close', start=0)
moving_avg(df2, 30, 'RSI', 'EMA_RSI', start=13)
df2.set_index('Date', inplace=True)


###############################################################################

''' Calculating CCI '''

df3 = pd.DataFrame.copy(df, deep=True)
df3.reset_index(inplace=True)

period = 20
df3['Typical_Price'] = (df3['High'] + df3['Low'] + df3['Close']) / 3
df3['SMA'] = df3['Typical_Price'].rolling(period).mean()
df3.fillna(0, inplace=True)
mean_deviation(df3, period)
df3.fillna(0, inplace=True)

df3['CCI']= ((df3['Typical_Price'] - df3['SMA']) / (0.015*df3['ADV']))
df3.set_index('Date', inplace=True)

###############################################################################


''' Plotting '''

#macd
fig1 = plt.figure()

ax1 = plt.subplot2grid((3,1),(0,0), rowspan=2)
ax1.plot(df1.index, df1['Close'], linewidth=0.5, gid='price', label='Price')
ax2 = plt.subplot2grid((3,1), (2,0), rowspan=1, sharex=ax1)
ax2.plot(df1.index, df1['MACD'].replace(0, np.nan), color='g', linewidth=0.5,
         gid='macd', label='MACD')
ax2.plot(df1.index, df1['Signal_Line'].replace(0, np.nan), color='r',
         linewidth=0.5, gid='signal', label='Signal Line')
p1=ax2.fill_between(df1.index, df1['Hist'], 0, color='g', where=df1['Hist']>0,
                 interpolate=True, alpha=0.5, gid='hist')
p2=ax2.fill_between(df1.index, df1['Hist'], 0, color='r', where=df1['Hist']<0,
                 interpolate=True, alpha=0.5, gid='hist')

macd_line_list = ['macd', 'signal']

ax1.set_title('Yes Bank Stocks', {'fontsize':12})
ax1.set_xlabel('Date')
ax1.set_ylabel('Prices in Rs')
ax1_l = ax1.legend(loc=2, prop={'size':10})

ax2.set_xlabel('Date')
ax2.set_ylabel('Prices in Rs')
ax2_l1 = ax2.legend(loc=2, ncol=2, prop={'size':10})
ax2_l1.get_frame().set_alpha(0.4)
ax2.add_artist(ax2_l1)
ax2_l2 = ax2.legend([(p1, p2)], ['Histogram'], numpoints=1,
               handler_map={tuple: HandlerTuple(ndivide=None)}, loc=3,
               prop={'size':10})
ax2_l2.get_frame().set_alpha(0.4)

plt.subplots_adjust(top=0.94, hspace=0.5)



#rsi
fig2 = plt.figure()
ax3 = plt.subplot2grid((3,1),(0,0), rowspan=2)

df2.reset_index(inplace=True)

cols = ['Date','Open','High','Low','Close','Volume']
ohlc = df2.loc[:, cols]
ohlc['Date'] = ohlc['Date'].map(mdates.date2num)
candlestick_ohlc(ax3, ohlc.values, width=10, colorup='g', colordown='r',
                 alpha=0.5)
ax3.plot(df2['Date'], df2['EMA_Close'].replace(0, np.nan), color='m',
         linewidth=0.7, gid='ema_close', label='EMA Close')

df2.set_index('Date', inplace=True)

ax4 = plt.subplot2grid((3,1),(2,0), sharex=ax3)
ax4.plot(df2['RSI'].replace(0, np.nan), color='r', linewidth=0.5, gid='rsi',
         label='RSI')
ax4.plot(df2['EMA_RSI'].replace(0, np.nan), color='m', linewidth=0.7,
         gid='ema_rsi', label='EMA RSI')
ax4.plot(df2.index, [70 for i in range(df2.shape[0])], color='k', linewidth=0.5)
ax4.plot(df2.index, [30 for i in range(df2.shape[0])], color='k', linewidth=0.5)
ax4.annotate('70', (df2.index[0],70), xycoords='data')
ax4.annotate('30', (df2.index[0],30), xycoords='data')

rsi_line_list = ['ema_close', 'rsi', 'ema_rsi']

ax3.set_title('Yes Bank Stocks', {'fontsize':12})
ax3.set_ylabel('Prices in Rs')
ax3_l = ax3.legend(loc=2, ncol=2, prop={'size':10})
ax3_l.get_frame().set_alpha(0.4)

ax4.set_xlabel('Date')
ax4.set_ylabel('Prices in Rs')
ax4_l = ax4.legend(loc=4, ncol=2, prop={'size':10})
ax4_l.get_frame().set_alpha(0.4)

plt.subplots_adjust(top=0.94, hspace=0.5)


#cci
fig3 = plt.figure()
ax5 = plt.subplot2grid((3,1), (0,0), rowspan=2)
ax5.plot(df3.index, df3['Close'], linewidth=0.5, gid='price', label='Price')

ax6 = plt.subplot2grid((3,1),(2,0), sharex=ax5)
ax6.plot(df3['CCI'], linewidth=0.5, gid='cci', label='CCI')
ax6.plot(df3.index, [100 for i in range(df3.shape[0])], color='m',
         linewidth=0.5)
ax6.plot(df3.index, [-100 for i in range(df3.shape[0])], color='m',
         linewidth=0.5)
ax6.plot(df3.index, [200 for i in range(df3.shape[0])], color='k',
         linewidth=0.5)
ax6.plot(df3.index, [-200 for i in range(df3.shape[0])], color='k',
         linewidth=0.5)

ax6.annotate('100', (df3.index[0],100), xycoords='data')
ax6.annotate('-100', (df3.index[0],-100), xycoords='data')
ax6.annotate('200', (df3.index[0],200), xycoords='data')
ax6.annotate('-200', (df3.index[0],-200), xycoords='data')

cci_line_list = ['cci']

ax5.set_title('Yes Bank Stocks', {'fontsize':12})
ax5.set_xlabel('Date')
ax5.set_ylabel('Prices in Rs')
ax5_l = ax5.legend(loc=2, ncol=2, prop={'size':10})
ax5_l.get_frame().set_alpha(0.4)

ax6.set_xlabel('Date')
ax6.set_ylabel('Prices in Rs')
ax6_l = ax6.legend(loc=7, ncol=2, prop={'size':10})
ax6_l.get_frame().set_alpha(0.4)

plt.subplots_adjust(top=0.94, hspace=0.5)


#cursor_funtionality objects
macd_ax1 = cursor_functionality.SnaptoCursor(ax1, df1,annotate_onplot=False)
macd_ax2 = cursor_functionality.SnaptoCursor(ax2, df1, annotate_onplot=True,
                                            line_list=macd_line_list)
rsi_ax3 = cursor_functionality.SnaptoCursor(ax3, df2, annotate_onplot=False)
rsi_ax4 = cursor_functionality.SnaptoCursor(ax4, df2, annotate_onplot=True,
                                            line_list=rsi_line_list)
cci_ax5 = cursor_functionality.SnaptoCursor(ax5, df3, annotate_onplot=False)
cci_ax6 = cursor_functionality.SnaptoCursor(ax6, df3, annotate_onplot=True,
                                            line_list=cci_line_list)


plt.show()


