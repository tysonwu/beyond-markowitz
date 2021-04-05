import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from collections import OrderedDict
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyfolio as pf


TOPN = 20
OBS_START = 'datetime(2016,1,1)' # string micmicking datetime object
BACKTEST_START = datetime(2017,1,1) # datetime
BACKTEST_END = '2019-12-31' # string


def empirical_estimate(data, tickers, start, end):
    start_dt = f'datetime({start[:4]},{int(start[5:7])},{int(start[8:10])})'
    end_dt = f'datetime({end[:4]},{int(end[5:7])},{int(end[8:10])})'
    ret = []
    for x, t in enumerate(tickers):
        df = data[t]
        df = df.query(f'Date >= {start_dt} and Date <= {end_dt}')
        ret.append(np.array((df['Logret']*100).fillna(0).to_list()))
    mean = np.matrix([sum(stock_ret) for stock_ret in ret]).transpose()
    cov = np.matrix(np.cov(ret))
    return ret, {'mean': mean, 'cov': cov}


def col_mat_to_list(z):
    return np.array((z/z.sum()).transpose())[0]


def gmv_portfolio(stat):
    covar = stat['cov']
    cov_inv = np.linalg.inv(covar)
    I = np.matrix(np.ones((len(tickers),1)))
    weights = cov_inv * I
    weights = col_mat_to_list(weights)
    return weights


def equal_portfolio(stat):
    return [1/len(tickers)]*len(tickers)


def gmv_no_short_portfolio(stat):
    covar = stat['cov']
    bnd = [(0, 1)]*len(tickers) # only positive weights
    w0 = [1/len(tickers)]*len(tickers)

    # min var optimization
    def calculate_portfolio_var(w,covar):
        w = np.matrix(w)
        return (w*covar*w.T)[0,0]

    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    weights = minimize(calculate_portfolio_var, 
    	w0, args=covar, bounds=bnd, method='SLSQP',constraints=cons)['x']
    return weights


def markowitz_no_short_portfolio(stat):
    mean = stat['mean']
    covar = stat['cov']
    
    w0 = [1/len(tickers)]*len(tickers)

    def calculate_cost(w, pars):
        mean = pars[0]
        covar = pars[1]
        w = np.matrix(w)
        return ((np.sqrt(w*covar*w.T)[0,0])*0.5 - (w*mean)[0,0])

    bnd = [(0, 1)]*len(tickers) # only positive weights
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    weights = minimize(calculate_cost, w0, args=[mean,covar], 
    	bounds=bnd, method='SLSQP',constraints=cons)
    return weights['x']


def erc_portfolio(stat):
    covar = stat['cov']

    def calculate_portfolio_var(w,V):
        w = np.matrix(w)
        return (w*V*w.T)[0,0]

    def calculate_risk_contribution(w,V):
        w = np.matrix(w)
        sigma = np.sqrt(calculate_portfolio_var(w,V))
        MRC = V*w.T
        RC = np.multiply(MRC,w.T)/sigma
        return RC

    def risk_budget_objective(x,pars):
        V = pars[0]
        x_t = pars[1]
        sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(sig_p,x_t))
        asset_RC = calculate_risk_contribution(x,V)
        J = sum(np.square(asset_RC-risk_target.T))[0,0] # sum of squared error
        return J

    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    
    w0 = [1/len(tickers)]*len(tickers)
    x_t = [1/len(tickers)]*len(tickers)
    cons = ({'type': 'eq', 'fun': total_weight_constraint},
     {'type': 'ineq', 'fun': long_only_constraint})
    weights = minimize(risk_budget_objective, w0, 
    	args=[covar,x_t], method='SLSQP',constraints=cons)
    return weights['x']


def mdp_portfolio(stat):
    covar = stat['cov']
    def calculate_portfolio_var(w,V):
        w = np.matrix(w)
        return (w*V*w.T)[0,0]

    def calc_diversification_ratio(w, covar):
        w_vol = np.dot(np.sqrt(np.diag(covar)), w.T)
        port_vol = np.sqrt(calculate_portfolio_var(w, covar))
        diversification_ratio = w_vol/port_vol
        return -diversification_ratio

    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x

    w0 = [1/len(tickers)]*len(tickers)
    cons = ({'type': 'eq', 'fun': total_weight_constraint}, 
    	{'type': 'ineq', 'fun': long_only_constraint})
    weights = minimize(calc_diversification_ratio, w0, 
    	args=covar, method='SLSQP', constraints=cons)
    return weights['x']


def weight_construction_portfolio(stat, mode):
    """
    specify method of weight_construction
    choices:
    gmv, markowitz, equal, gmv_no_short, markowitz_no_short, erc
    """
    if mode == 'gmv':
        return gmv_portfolio(stat)
    if mode == 'equal':
        return equal_portfolio(stat)
    if mode == 'gmv_no_short':
        return gmv_no_short_portfolio(stat)
    if mode == 'markowitz_no_short':
        return markowitz_no_short_portfolio(stat)
    if mode == 'erc':
        return erc_portfolio(stat)
    if mode == 'mdp':
        return mdp_portfolio(stat)


def backtest(mode):
    for m in mode:
        weights_backtest = {}
        for i in tqdm(range(37)): # 37 for 2019-01-01
            observation_start_date = datetime.strftime(
            	BACKTEST_START+relativedelta(months=i)-relativedelta(years=1), 
            	'%Y-%m-%d')
            rebalance_date = datetime.strftime(
            	BACKTEST_START+relativedelta(months=i), '%Y-%m-%d')
            _, stat = empirical_estimate(data, 
            	tickers, 
            	observation_start_date, 
            	rebalance_date)
            weights_backtest[rebalance_date] = weight_construction_portfolio(stat, m)
            # print(weights_backtest[rebalance_date])

        datemap = OrderedDict(weights_backtest)
        # print(datemap)
        # get all trading dates
        df_backtest = pd.DataFrame({'Date': list(data['MSFT'].index)})
        df_backtest = df_backtest[df_backtest['Date']>=datetime.strftime(
        	BACKTEST_START,'%Y-%m-%d')]
        df_backtest = df_backtest[df_backtest['Date']<=BACKTEST_END]
        df_backtest['Date'] = df_backtest['Date'].apply(
        	lambda x: datetime.strftime(x,'%Y-%m-%d'))

        def find_weight(dt):
            result = None
            for dm, w in datemap.items():
                if dt >= dm:
                    result = w
                if dt < dm:
                    return result
            else:
                return result

        df_backtest['Weight'] = df_backtest['Date'].apply(find_weight)
        df_backtest['Logret'] = df_backtest['Date'].apply(
        	lambda x: [data[t].loc[x,'Logret'] for t in tickers])
        df_backtest['Daily_Logret'] = df_backtest.apply(
        	lambda x: sum(r*w for r,w in zip(x['Weight'],x['Logret'])) , axis=1)

        # Benchmark viz
        df_backtest['Date_dt'] = df_backtest['Date'].apply(
        	lambda x: datetime.strptime(x, '%Y-%m-%d'))
        plt.plot(df_backtest['Date_dt'],df_backtest['Daily_Logret'].cumsum(),label=m)

        #Performance metrics
        if len(mode) == 1:
            df_metrics = df_backtest[['Date_dt','Daily_Logret']]
            df_metrics = df_metrics.set_index('Date_dt')
            pf.create_simple_tear_sheet(df_metrics['Daily_Logret'])
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # obtain data and preprocessing
    nasdaq_data = pd.read_csv('./data/nasdaq_stocks.csv')
    mc = {row['Symbol']: row['Market Cap'] for _, row in nasdaq_data.iterrows() \
     if row['Market Cap'] != 0 and not np.isnan(row['Market Cap'])}
    mcdf = pd.DataFrame.from_dict(mc, orient='index', columns=['marketcap'])
    mcdf_chosen = mcdf.nlargest(TOPN,'marketcap', keep='first')
    tickers = mcdf_chosen.index.to_list()
    data = yf.download(
    	tickers=' '.join(tickers) , 
    	period='max', 
    	interval='1d', 
    	threads=True, 
    	group_by = 'ticker')
    data = {t:data[t] for t in tickers}

    for v in data.values():
        v['Return'] = v['Close']/v['Close'].shift(1)
        v['Logret'] = v['Return'].apply(lambda x: np.log(x))
    data = {k:v.query(f'Date >= {OBS_START}') for k,v in data.items()}

    # generate weights and backtesting
    backtest(mode=['mdp','markowitz_no_short','gmv','equal','erc','gmv_no_short'])

