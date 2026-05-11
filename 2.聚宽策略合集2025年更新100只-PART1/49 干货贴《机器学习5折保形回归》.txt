# 风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
# 原文一般包含策略说明，如有疑问请到原文和作者交流讨论。
# 原文网址：https://www.joinquant.com/view/community/detail/48114
# 标题：干货贴《机器学习5折保形回归》
# 作者：MarioC

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import sklearn
from six import StringIO,BytesIO # py3的环境，使用BytesIO
from xgboost import XGBClassifier ,XGBRegressor
from sklearn.model_selection import KFold
# 初始化函数
def initialize(context):
    # 设定基准
    set_benchmark('000985.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化全局变量
    g.no_trading_today_signal = False
    g.stock_num = 5
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    

    df1 = pd.read_csv(BytesIO(read_file('train_conformal_base.csv'))).dropna()
    df2 = pd.read_csv(BytesIO(read_file('test_conformal_base.csv'))).dropna()
    X_train = pd.concat([df1, df2], axis=0)
    
    y = X_train['LABEL']
    g.factor_list = ['non_linear_size',#非线性市值(风格因子)
                'beta',            #贝塔
                 'book_to_price_ratio',  #市面账值比
                'earnings_yield',  #盈利能力，
               'growth'            #成长 
               ]
    X = X_train[g.factor_list]
    
    
    # 定义五折交叉验证
    kf = KFold(n_splits=5, shuffle=False, random_state=42)
    
    # 遍历每一折
    for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}:")
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # 建立模型和保形回归器  
        model = XGBRegressor()  # 建立预测模型
        nc = NcFactory.create_nc(model)  # 建立不符合度量
        icp = IcpRegressor(nc)  # 建立一个归纳保形回归器
        
        # 拟合和校准模型  
        icp.fit(X_train.values, y_train.values)
        icp.calibrate(X_valid.values, y_valid.values)
        
        # 将模型保存到相应的属性中
        setattr(g, f"models{fold+1}", icp)

    
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_monthly(weekly_adjustment, 1, '9:30')
    run_daily(check_limit_up, '14:00') 
    run_daily(close_account, '14:30')


            
# 1-1 准备股票池
def prepare_stock_list(context):
    # 获取已持有列表
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    # 获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close', 'high_limit'],
                       count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
        
    
# 1-2 选股模块
def get_stock_list(context):
    # 指定日期防止未来数据
    yesterday = context.previous_date
    today = context.current_dt
    # 获取初始列表
    stocks = get_all_securities('stock', yesterday).index.tolist()
    initial_list = filter_kcbj_stock(stocks)
    initial_list = filter_st_stock(initial_list)
    initial_list = filter_paused_stock(initial_list)
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_limitup_stock(context,initial_list)
    initial_list = filter_limitdown_stock(context,initial_list)
    factor_data = get_factor_values(initial_list, g.factor_list, end_date=yesterday, count=1)
    df_jq_factor_value = pd.DataFrame(index=initial_list, columns=g.factor_list)
    for factor in g.factor_list:
        df_jq_factor_value[factor] = list(factor_data[factor].T.iloc[:, 0])
    df_jq_factor_value = data_preprocessing(df_jq_factor_value, initial_list, industry_code, yesterday)
    
    X=df_jq_factor_value.values
    Y1 = g.models1.predict(X, significance=0.05)
    Y2 = g.models2.predict(X, significance=0.05)
    Y3 = g.models3.predict(X, significance=0.05)
    Y4 = g.models4.predict(X, significance=0.05)
    Y5 = g.models5.predict(X, significance=0.05)
    Y=Y1+Y2+Y3+Y4+Y5
    y_lower = Y[:, 0]  
    y_upper = Y[:, 1]  
    y_pred = (y_lower + y_upper) / 2  

    
    tar = y_pred
    df = df_jq_factor_value
    df['total_score'] = list(tar)
    df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
    lst = df.index.tolist()
    lst = lst[:min(g.stock_num, len(lst))]
    return lst


# 1-3 整体调整持仓
def weekly_adjustment(context):
    # 获取应买入列表
    target_list = get_stock_list(context)
    # 调仓卖出
    for stock in g.hold_list:
        if (stock not in target_list) and (stock not in g.yesterday_HL_list):
            position = context.portfolio.positions[stock]
            close_position(position)
    position_count = len(context.portfolio.positions)
    target_num = len(target_list)
    if target_num > position_count:
        value = context.portfolio.cash / (target_num - position_count)
        for stock in target_list:
            if stock not in list(context.portfolio.positions.keys()):
                if open_position(stock, value):
                    if len(context.portfolio.positions) == target_num:
                        break



# 1-4 调整昨日涨停股票
def check_limit_up(context):
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        # 对昨日涨停股票观察到尾盘如不涨停则提前卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("[%s]涨停，继续持有" % (stock))

# 3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)


# 3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False


# 3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


# 4-1 判断今天是否为账户资金再平衡的日期
def today_is_between(context, start_date, end_date):
    today = context.current_dt.strftime('%m-%d')
    if (start_date <= today) and (today <= end_date):
        return True
    else:
        return False


# 4-2 清仓后次日资金可转
def close_account(context):
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0:
            for stock in g.hold_list:
                position = context.portfolio.positions[stock]
                close_position(position)
                log.info("卖出[%s]" % (stock))

# 2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


# 2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[0] == '3':
            stock_list.remove(stock)
    return stock_list


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if
            not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]



def get_industry_name(i_Constituent_Stocks, value):
    return [k for k, v in i_Constituent_Stocks.items() if value in v]

#缺失值处理
def replace_nan_indu(factor_data,stockList,industry_code,date):
    #把nan用行业平均值代替，依然会有nan，此时用所有股票平均值代替
    i_Constituent_Stocks={}
    data_temp=pd.DataFrame(index=industry_code,columns=factor_data.columns)
    for i in industry_code:
        temp = get_industry_stocks(i, date)
        i_Constituent_Stocks[i] = list(set(temp).intersection(set(stockList)))
        data_temp.loc[i]=mean(factor_data.loc[i_Constituent_Stocks[i],:])
    for factor in data_temp.columns:
        #行业缺失值用所有行业平均值代替
        null_industry=list(data_temp.loc[pd.isnull(data_temp[factor]),factor].keys())
        for i in null_industry:
            data_temp.loc[i,factor]=mean(data_temp[factor])
        null_stock=list(factor_data.loc[pd.isnull(factor_data[factor]),factor].keys())
        for i in null_stock:
            industry=get_industry_name(i_Constituent_Stocks, i)
            if industry:
                factor_data.loc[i,factor]=data_temp.loc[industry[0],factor] 
            else:
                factor_data.loc[i,factor]=mean(factor_data[factor])
    return factor_data
industry_code = ['801010','801020','801030','801040','801050','801080','801110','801120','801130','801140','801150',\
                    '801160','801170','801180','801200','801210','801230','801710','801720','801730','801740','801750',\
                   '801760','801770','801780','801790','801880','801890']
def data_preprocessing(factor_data,stockList,industry_code,date):
    #去极值
    factor_data=winsorize_med(factor_data, scale=5, inf2nan=False,axis=0)
    #缺失值处理
    factor_data=replace_nan_indu(factor_data,stockList,industry_code,date)
    factor_data=standardlize(factor_data,axis=0)
    return factor_data

    
        
#定义base

"""
docstring
"""

# Authors: Henrik Linusson

import abc
import numpy as np

from sklearn.base import BaseEstimator


class RegressorMixin(object):
	def __init__(self):
		super(RegressorMixin, self).__init__()

	@classmethod
	def get_problem_type(cls):
		return 'regression'


class ClassifierMixin(object):
	def __init__(self):
		super(ClassifierMixin, self).__init__()

	@classmethod
	def get_problem_type(cls):
		return 'classification'


class BaseModelAdapter(BaseEstimator):
	__metaclass__ = abc.ABCMeta

	def __init__(self, model, fit_params=None):
		super(BaseModelAdapter, self).__init__()

		self.model = model
		self.last_x, self.last_y = None, None
		self.clean = False
		self.fit_params = {} if fit_params is None else fit_params

	def fit(self, x, y):
		"""Fits the model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the model.

		Returns
		-------
		None
		"""

		self.model.fit(x, y, **self.fit_params)
		self.clean = False

	def predict(self, x):
		"""Returns the prediction made by the underlying model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of test examples.

		Returns
		-------
		y : numpy array of shape [n_samples]
			Predicted outputs of test examples.
		"""
		if (
			not self.clean or
			self.last_x is None or
			self.last_y is None or
			not np.array_equal(self.last_x, x)
		):
			self.last_x = x
			self.last_y = self._underlying_predict(x)
			self.clean = True

		return self.last_y.copy()

	@abc.abstractmethod
	def _underlying_predict(self, x):
		"""Produces a prediction using the encapsulated model.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of test examples.

		Returns
		-------
		y : numpy array of shape [n_samples]
			Predicted outputs of test examples.
		"""
		pass


class ClassifierAdapter(BaseModelAdapter):
	def __init__(self, model, fit_params=None):
		super(ClassifierAdapter, self).__init__(model, fit_params)

	def _underlying_predict(self, x):
		return self.model.predict_proba(x)


class RegressorAdapter(BaseModelAdapter):
	def __init__(self, model, fit_params=None):
		super(RegressorAdapter, self).__init__(model, fit_params)

	def _underlying_predict(self, x):
		return self.model.predict(x)


class OobMixin(object):
	def __init__(self, model, fit_params=None):
		super(OobMixin, self).__init__(model, fit_params)
		self.train_x = None

	def fit(self, x, y):
		super(OobMixin, self).fit(x, y)
		self.train_x = x

	def _underlying_predict(self, x):
		# TODO: sub-sampling of ensemble for test patterns
		oob = x == self.train_x

		if hasattr(oob, 'all'):
			oob = oob.all()

		if oob:
			return self._oob_prediction()
		else:
			return super(OobMixin, self)._underlying_predict(x)


class OobClassifierAdapter(OobMixin, ClassifierAdapter):
	def __init__(self, model, fit_params=None):
		super(OobClassifierAdapter, self).__init__(model, fit_params)

	def _oob_prediction(self):
		return self.model.oob_decision_function_


class OobRegressorAdapter(OobMixin, RegressorAdapter):
	def __init__(self, model, fit_params=None):
		super(OobRegressorAdapter, self).__init__(model, fit_params)

	def _oob_prediction(self):
		return self.model.oob_prediction_
		
#定义BaseScorer
class BaseScorer(sklearn.base.BaseEstimator):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(BaseScorer, self).__init__()

	@abc.abstractmethod
	def fit(self, x, y):
		pass

	@abc.abstractmethod
	def score(self, x, y=None):
		pass


class RegressorNormalizer(BaseScorer):
	def __init__(self, base_model, normalizer_model, err_func):
		super(RegressorNormalizer, self).__init__()
		self.base_model = base_model
		self.normalizer_model = normalizer_model
		self.err_func = err_func

	def fit(self, x, y):
		residual_prediction = self.base_model.predict(x)
		residual_error = np.abs(self.err_func.apply(residual_prediction, y))
		residual_error += 0.00001 # Add small term to avoid log(0)
		log_err = np.log(residual_error)
		self.normalizer_model.fit(x, log_err)

	def score(self, x, y=None):
		norm = np.exp(self.normalizer_model.predict(x))
		return norm
#定义BaseModelNc
class BaseModelNc(BaseScorer):
	"""Base class for nonconformity scorers based on an underlying model.

	Parameters
	----------
	model : ClassifierAdapter or RegressorAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc or RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.
	"""
	def __init__(self, model, err_func, normalizer=None, beta=0):
		super(BaseModelNc, self).__init__()
		self.err_func = err_func
		self.model = model
		self.normalizer = normalizer
		self.beta = beta

		# If we use sklearn.base.clone (e.g., during cross-validation),
		# object references get jumbled, so we need to make sure that the
		# normalizer has a reference to the proper model adapter, if applicable.
		if (self.normalizer is not None and
			hasattr(self.normalizer, 'base_model')):
			self.normalizer.base_model = self.model

		self.last_x, self.last_y = None, None
		self.last_prediction = None
		self.clean = False

	def fit(self, x, y):
		"""Fits the underlying model of the nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying model.

		Returns
		-------
		None
		"""
		self.model.fit(x, y)
		if self.normalizer is not None:
			self.normalizer.fit(x, y)
		self.clean = False

	def score(self, x, y=None):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		prediction = self.model.predict(x)
		n_test = x.shape[0]
		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		return self.err_func.apply(prediction, y) / norm


#定义TcpClassifier
class TcpClassifier(BaseEstimator, ClassifierMixin):
	"""Transductive conformal classifier.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``
		and ``calc_nc(x, y)``.

	smoothing : boolean
		Decides whether to use stochastic smoothing of p-values.

	Attributes
	----------
	train_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of training set.

	train_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	classes : numpy array of shape [n_classes]
		List of class labels, with indices corresponding to output columns
		 of TcpClassifier.predict()

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning
	in a random world. Springer Science & Business Media.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.svm import SVC
	>>> from nonconformist.base import ClassifierAdapter
	>>> from nonconformist.cp import TcpClassifier
	>>> from nonconformist.nc import ClassifierNc, MarginErrFunc
	>>> iris = load_iris()
	>>> idx = np.random.permutation(iris.target.size)
	>>> train = idx[:int(idx.size / 2)]
	>>> test = idx[int(idx.size / 2):]
	>>> model = ClassifierAdapter(SVC(probability=True))
	>>> nc = ClassifierNc(model, MarginErrFunc())
	>>> tcp = TcpClassifier(nc)
	>>> tcp.fit(iris.data[train, :], iris.target[train])
	>>> tcp.predict(iris.data[test, :], significance=0.10)
	...             # doctest: +SKIP
	array([[ True, False, False],
		[False,  True, False],
		...,
		[False,  True, False],
		[False,  True, False]], dtype=bool)
	"""

	def __init__(self, nc_function, condition=None, smoothing=True):
		self.train_x, self.train_y = None, None
		self.nc_function = nc_function
		super(TcpClassifier, self).__init__()

		# Check if condition-parameter is the default function (i.e.,
		# lambda x: 0). This is so we can safely clone the object without
		# the clone accidentally having self.conditional = True.
		default_condition = lambda x: 0
		is_default = (callable(condition) and
		              (condition.__code__.co_code ==
		               default_condition.__code__.co_code))

		if is_default:
			self.condition = condition
			self.conditional = False
		elif callable(condition):
			self.condition = condition
			self.conditional = True
		else:
			self.condition = lambda x: 0
			self.conditional = False

		self.smoothing = smoothing

		self.base_icp = IcpClassifier(
			self.nc_function,
			self.condition,
			self.smoothing
		)

		self.classes = None

	def fit(self, x, y):
		self.train_x, self.train_y = x, y
		self.classes = np.unique(y)

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float or None
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then the p-values
			are output rather than the predictions.

		Returns
		-------
		p : numpy array of shape [n_samples, n_classes]
			If significance is ``None``, then p contains the p-values for each
			sample-class pair; if significance is a float between 0 and 1, then
			p is a boolean array denoting which labels are included in the
			prediction sets.
		"""
		n_test = x.shape[0]
		n_train = self.train_x.shape[0]
		p = np.zeros((n_test, self.classes.size))
		for i in range(n_test):
			for j, y in enumerate(self.classes):
				train_x = np.vstack([self.train_x, x[i, :]])
				train_y = np.hstack([self.train_y, y])
				self.base_icp.fit(train_x, train_y)
				self.base_icp.calibrate(train_x, train_y)

				ncal_ngt_neq = self.base_icp._get_stats(x[i, :].reshape(1, x.shape[1]))

				ncal = ncal_ngt_neq[:, j, 0]
				ngt = ncal_ngt_neq[:, j, 1]
				neq = ncal_ngt_neq[:, j, 2]

				p[i, j] = calc_p(ncal - 1, ngt, neq - 1, self.smoothing)

		if significance is not None:
			return p > significance
		else:
			return p

	def predict_conf(self, x):
		"""Predict the output values for a set of input patterns, using
		the confidence-and-credibility output scheme.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		Returns
		-------
		p : numpy array of shape [n_samples, 3]
			p contains three columns: the first column contains the most
			likely class for each test pattern; the second column contains
			the confidence in the predicted class label, and the third column
			contains the credibility of the prediction.
		"""
		p = self.predict(x, significance=None)
		label = p.argmax(axis=1)
		credibility = p.max(axis=1)
		for i, idx in enumerate(label):
			p[i, idx] = -np.inf
		confidence = 1 - p.max(axis=1)

		return np.array([label, confidence, credibility]).T
		
#定义NcFactory
class NcFactory(object):
	@staticmethod
	def create_nc(model, err_func=None, normalizer_model=None, oob=False,
                      fit_params=None, fit_params_normalizer=None):
		if normalizer_model is not None:
			normalizer_adapter = RegressorAdapter(normalizer_model, fit_params_normalizer)
		else:
			normalizer_adapter = None

		if isinstance(model, sklearn.base.ClassifierMixin):
			err_func = MarginErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_decision_function_'):
					adapter = OobClassifierAdapter(model, fit_params)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                      'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = ClassifierAdapter(model, fit_params)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return ClassifierNc(adapter, err_func, normalizer)
			else:
				return ClassifierNc(adapter, err_func)

		elif isinstance(model, sklearn.base.RegressorMixin):
			err_func = AbsErrorErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_prediction_'):
					adapter = OobRegressorAdapter(model, fit_params)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                     'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = RegressorAdapter(model, fit_params)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return RegressorNc(adapter, err_func, normalizer)
			else:
				return RegressorNc(adapter, err_func)
#定义TcpClassifier
class TcpClassifier(BaseEstimator, ClassifierMixin):
	"""Transductive conformal classifier.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``
		and ``calc_nc(x, y)``.

	smoothing : boolean
		Decides whether to use stochastic smoothing of p-values.

	Attributes
	----------
	train_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of training set.

	train_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	classes : numpy array of shape [n_classes]
		List of class labels, with indices corresponding to output columns
		 of TcpClassifier.predict()

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic learning
	in a random world. Springer Science & Business Media.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.svm import SVC
	>>> from nonconformist.base import ClassifierAdapter
	>>> from nonconformist.cp import TcpClassifier
	>>> from nonconformist.nc import ClassifierNc, MarginErrFunc
	>>> iris = load_iris()
	>>> idx = np.random.permutation(iris.target.size)
	>>> train = idx[:int(idx.size / 2)]
	>>> test = idx[int(idx.size / 2):]
	>>> model = ClassifierAdapter(SVC(probability=True))
	>>> nc = ClassifierNc(model, MarginErrFunc())
	>>> tcp = TcpClassifier(nc)
	>>> tcp.fit(iris.data[train, :], iris.target[train])
	>>> tcp.predict(iris.data[test, :], significance=0.10)
	...             # doctest: +SKIP
	array([[ True, False, False],
		[False,  True, False],
		...,
		[False,  True, False],
		[False,  True, False]], dtype=bool)
	"""

	def __init__(self, nc_function, condition=None, smoothing=True):
		self.train_x, self.train_y = None, None
		self.nc_function = nc_function
		super(TcpClassifier, self).__init__()

		# Check if condition-parameter is the default function (i.e.,
		# lambda x: 0). This is so we can safely clone the object without
		# the clone accidentally having self.conditional = True.
		default_condition = lambda x: 0
		is_default = (callable(condition) and
		              (condition.__code__.co_code ==
		               default_condition.__code__.co_code))

		if is_default:
			self.condition = condition
			self.conditional = False
		elif callable(condition):
			self.condition = condition
			self.conditional = True
		else:
			self.condition = lambda x: 0
			self.conditional = False

		self.smoothing = smoothing

		self.base_icp = IcpClassifier(
			self.nc_function,
			self.condition,
			self.smoothing
		)

		self.classes = None

	def fit(self, x, y):
		self.train_x, self.train_y = x, y
		self.classes = np.unique(y)

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float or None
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then the p-values
			are output rather than the predictions.

		Returns
		-------
		p : numpy array of shape [n_samples, n_classes]
			If significance is ``None``, then p contains the p-values for each
			sample-class pair; if significance is a float between 0 and 1, then
			p is a boolean array denoting which labels are included in the
			prediction sets.
		"""
		n_test = x.shape[0]
		n_train = self.train_x.shape[0]
		p = np.zeros((n_test, self.classes.size))
		for i in range(n_test):
			for j, y in enumerate(self.classes):
				train_x = np.vstack([self.train_x, x[i, :]])
				train_y = np.hstack([self.train_y, y])
				self.base_icp.fit(train_x, train_y)
				self.base_icp.calibrate(train_x, train_y)

				ncal_ngt_neq = self.base_icp._get_stats(x[i, :].reshape(1, x.shape[1]))

				ncal = ncal_ngt_neq[:, j, 0]
				ngt = ncal_ngt_neq[:, j, 1]
				neq = ncal_ngt_neq[:, j, 2]

				p[i, j] = calc_p(ncal - 1, ngt, neq - 1, self.smoothing)

		if significance is not None:
			return p > significance
		else:
			return p

	def predict_conf(self, x):
		"""Predict the output values for a set of input patterns, using
		the confidence-and-credibility output scheme.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		Returns
		-------
		p : numpy array of shape [n_samples, 3]
			p contains three columns: the first column contains the most
			likely class for each test pattern; the second column contains
			the confidence in the predicted class label, and the third column
			contains the credibility of the prediction.
		"""
		p = self.predict(x, significance=None)
		label = p.argmax(axis=1)
		credibility = p.max(axis=1)
		for i, idx in enumerate(label):
			p[i, idx] = -np.inf
		confidence = 1 - p.max(axis=1)

		return np.array([label, confidence, credibility]).T
		
#定义BaseIcp
class BaseIcp(BaseEstimator):
	"""Base class for inductive conformal predictors.
	"""
	def __init__(self, nc_function, condition=None):
		self.cal_x, self.cal_y = None, None
		self.nc_function = nc_function

		# Check if condition-parameter is the default function (i.e.,
		# lambda x: 0). This is so we can safely clone the object without
		# the clone accidentally having self.conditional = True.
		default_condition = lambda x: 0
		is_default = (callable(condition) and
		              (condition.__code__.co_code ==
		               default_condition.__code__.co_code))

		if is_default:
			self.condition = condition
			self.conditional = False
		elif callable(condition):
			self.condition = condition
			self.conditional = True
		else:
			self.condition = lambda x: 0
			self.conditional = False

	def fit(self, x, y):
		"""Fit underlying nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the nonconformity scorer.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the nonconformity scorer.

		Returns
		-------
		None
		"""
		# TODO: incremental?
		self.nc_function.fit(x, y)

	def calibrate(self, x, y, increment=False):
		"""Calibrate conformal predictor based on underlying nonconformity
		scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for calibrating the conformal predictor.

		y : numpy array of shape [n_samples, n_features]
			Outputs of examples for calibrating the conformal predictor.

		increment : boolean
			If ``True``, performs an incremental recalibration of the conformal
			predictor. The supplied ``x`` and ``y`` are added to the set of
			previously existing calibration examples, and the conformal
			predictor is then calibrated on both the old and new calibration
			examples.

		Returns
		-------
		None
		"""
		self._calibrate_hook(x, y, increment)
		self._update_calibration_set(x, y, increment)

		if self.conditional:
			category_map = np.array([self.condition((x[i, :], y[i]))
									 for i in range(y.size)])
			self.categories = np.unique(category_map)
			self.cal_scores = defaultdict(partial(np.ndarray, 0))

			for cond in self.categories:
				idx = category_map == cond
				cal_scores = self.nc_function.score(self.cal_x[idx, :],
				                                    self.cal_y[idx])
				self.cal_scores[cond] = np.sort(cal_scores)[::-1]
		else:
			self.categories = np.array([0])
			cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
			self.cal_scores = {0: np.sort(cal_scores)[::-1]}

	def _calibrate_hook(self, x, y, increment):
		pass

	def _update_calibration_set(self, x, y, increment):
		if increment and self.cal_x is not None and self.cal_y is not None:
			self.cal_x = np.vstack([self.cal_x, x])
			self.cal_y = np.hstack([self.cal_y, y])
		else:
			self.cal_x, self.cal_y = x, y
			
#定义 IcpRegressor
class IcpRegressor(BaseIcp, RegressorMixin):
	"""Inductive conformal regressor.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``,
		``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

	Attributes
	----------
	cal_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of calibration set.

	cal_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
		Inductive confidence machines for regression. In Machine Learning: ECML
		2002 (pp. 345-356). Springer Berlin Heidelberg.

	.. [2] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
		intervals with regression neural networks. Neural Networks, 24(8),
		842-851.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_boston
	>>> from sklearn.tree import DecisionTreeRegressor
	>>> from nonconformist.base import RegressorAdapter
	>>> from nonconformist.icp import IcpRegressor
	>>> from nonconformist.nc import RegressorNc, AbsErrorErrFunc
	>>> boston = load_boston()
	>>> idx = np.random.permutation(boston.target.size)
	>>> train = idx[:int(idx.size / 3)]
	>>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
	>>> test = idx[int(2 * idx.size / 3):]
	>>> model = RegressorAdapter(DecisionTreeRegressor())
	>>> nc = RegressorNc(model, AbsErrorErrFunc())
	>>> icp = IcpRegressor(nc)
	>>> icp.fit(boston.data[train, :], boston.target[train])
	>>> icp.calibrate(boston.data[cal, :], boston.target[cal])
	>>> icp.predict(boston.data[test, :], significance=0.10)
	...     # doctest: +SKIP
	array([[  5. ,  20.6],
		[ 15.5,  31.1],
		...,
		[ 14.2,  29.8],
		[ 11.6,  27.2]])
	"""
	def __init__(self, nc_function, condition=None):
		super(IcpRegressor, self).__init__(nc_function, condition)

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99}
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		# TODO: interpolated p-values

		n_significance = (99 if significance is None
		                  else np.array(significance).size)

		if n_significance > 1:
			prediction = np.zeros((x.shape[0], 2, n_significance))
		else:
			prediction = np.zeros((x.shape[0], 2))

		condition_map = np.array([self.condition((x[i, :], None))
		                          for i in range(x.shape[0])])

		for condition in self.categories:
			idx = condition_map == condition
			if np.sum(idx) > 0:
				p = self.nc_function.predict(x[idx, :],
				                             self.cal_scores[condition],
				                             significance)
				if n_significance > 1:
					prediction[idx, :, :] = p
				else:
					prediction[idx, :] = p

		return prediction
		
#定义RegressionErrFunc
class RegressionErrFunc(object):
	"""Base class for regression model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(RegressionErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):#, norm=None, beta=0):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass

	@abc.abstractmethod
	def apply_inverse(self, nc, significance):#, norm=None, beta=0):
		"""Apply the inverse of the nonconformity function (i.e.,
		calculate prediction interval).

		Parameters
		----------
		nc : numpy array of shape [n_calibration_samples]
			Nonconformity scores obtained for conformal predictor.

		significance : float
			Significance level (0, 1).

		Returns
		-------
		interval : numpy array of shape [n_samples, 2]
			Minimum and maximum interval boundaries for each prediction.
		"""
		pass

#定义AbsErrorErrFunc
class AbsErrorErrFunc(RegressionErrFunc):
	"""Calculates absolute error nonconformity for regression problems.

		For each correct output in ``y``, nonconformity is defined as

		.. math::
			| y_i - \hat{y}_i |
	"""

	def __init__(self):
		super(AbsErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return np.abs(prediction - y)

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([nc[border], nc[border]])
		
#定义RegressorNc
class RegressorNc(BaseModelNc):
	"""Nonconformity scorer using an underlying regression model.

	Parameters
	----------
	model : RegressorAdapter
		Underlying regression model used for calculating nonconformity scores.

	err_func : RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : RegressorAdapter
		Underlying model object.

	err_func : RegressionErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	ProbEstClassifierNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
	             err_func=AbsErrorErrFunc(),
	             normalizer=None,
	             beta=0):
		super(RegressorNc, self).__init__(model,
		                                  err_func,
		                                  normalizer,
		                                  beta)

	def predict(self, x, nc, significance=None):
		"""Constructs prediction intervals for a set of test examples.

		Predicts the output of each test pattern using the underlying model,
		and applies the (partial) inverse nonconformity function to each
		prediction, resulting in a prediction interval for each test pattern.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		n_test = x.shape[0]
		prediction = self.model.predict(x)
		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		if significance:
			intervals = np.zeros((x.shape[0], 2))
			err_dist = self.err_func.apply_inverse(nc, significance)
			err_dist = np.hstack([err_dist] * n_test)
			err_dist *= norm

			intervals[:, 0] = prediction - err_dist[0, :]
			intervals[:, 1] = prediction + err_dist[1, :]

			return intervals
		else:
			significance = np.arange(0.01, 1.0, 0.01)
			intervals = np.zeros((x.shape[0], 2, significance.size))

			for i, s in enumerate(significance):
				err_dist = self.err_func.apply_inverse(nc, s)
				err_dist = np.hstack([err_dist] * n_test)
				err_dist *= norm

				intervals[:, 0, i] = prediction - err_dist[0, :]
				intervals[:, 1, i] = prediction + err_dist[0, :]

			return intervals