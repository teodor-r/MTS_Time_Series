import seaborn as sns
from sktime.forecasting.fbprophet import Prophet
import statsmodels.api as sm
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import MeanAbsoluteError,MeanAbsolutePercentageError
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sktime.forecasting.fbprophet import Prophet as prophet_sktime
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.base import ForecastingHorizon
from scipy import fft
from statsmodels.graphics import tsaplots
from sklearn.ensemble import IsolationForest
from sktime.forecasting.trend import STLForecaster
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA, StatsForecastAutoETS, StatsForecastMSTL
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TimeSeriesPredict():
    def __init__(self, sales, sales_dates, sales_prices, store_id):
        self.store_id = store_id
        self.sales = sales
        self.store = self.preprocess_df(sales_dates, sales_prices)
        self.sales_dates = sales_dates
        self.items_names = self.store.item_id.unique()
        self.h_forecasts = [7, 30, 90]

    def preprocess_df(self, sales_dates, sales_prices):
        """
        Cобирает агреггированную информацию для заданного магазина
        """
        new_merged = sales_dates.merge(sales_prices, how='inner', on="wm_yr_wk")
        columns_to_keep = ['date', 'wm_yr_wk', 'wday', 'month', 'year', 'date_id', 'item_id', 'sell_price', 'store_id']
        merged_data = new_merged[columns_to_keep]
        return merged_data[merged_data.store_id == self.store_id]

    def get_item_data(self, name):
        """
        Выбирает данные для заданного товара, заменяет индексы на метки времени
        """
        without_time_index = self.store[(self.store.item_id == name)]
        without_time_index['date_id'] = without_time_index['date_id'].astype(int)
        merged = without_time_index.merge(self.sales_dates[['date_id', 'date']], how='inner', on='date_id')
        merged.date = merged.date.astype(str)
        merged.date = pd.to_datetime(merged.date)
        merged.set_index('date', inplace=True)
        merged.drop(['store_id', 'item_id', 'date_id'], axis=1, inplace=True)
        merged.columns = ['y']
        return merged

    def get_item_data_boost(self, name):
        """
        Выбирает данные для заданного товара, заменяет индексы на метки времени
        """
        without_time_index = self.store[(self.store.item_id == name)]
        without_time_index['date_id'] = without_time_index['date_id'].astype(int)
        merged = without_time_index.merge(self.sales[['cnt', 'date_id', 'item_id']], how='inner',
                                          on=['date_id', 'item_id'])
        merged.date = merged.date.astype(str)
        merged.date = pd.to_datetime(merged.date)
        merged.set_index('date', inplace=True)
        merged.drop(['store_id', 'item_id', 'date_id', "wm_yr_wk"], axis=1, inplace=True)
        merged.rename(columns={'cnt': 'y'}, inplace=True)
        return merged

    def smape(self, y_true, y_pred):
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        return (numerator / denominator) * 100

    def BC_inverse(self, y, lam, sig):
        """Back transform Box-Cox with Bias correction.
        See https://robjhyndman.com/hyndsight/backtransforming
        """
        if lam == 0:
            return np.exp(y) * (1 + 0.5 * sig ** 2)
        else:
            res = np.power(lam * y + 1., 1 / lam)
            res *= (1 + 0.5 * sig ** 2 * (1 - lam) / (lam * y + 1.) ** 2)
            return res

    def metrics_report(self, y_true, y_pred):

        """
        Функция оценки качества прогноза. Оценивает персентили распределения ошибки
        (медиану, среднее, 3й квартиль и 95-персентиль)
        по абсолютной и относительной процентной ошибки для прогноза с помощью модели (y_pred) и
        для базового прогноза, равному среднему значению вр.

        """

        mape_func = MeanAbsolutePercentageError()
        mae_func = MeanAbsoluteError()

        y_true_mean = y_true.mean() * np.ones(shape=(len(y_true),))

        mapes = mape_func.evaluate_by_index(y_true, y_pred) * 100
        maes = mae_func.evaluate_by_index(y_true, y_pred)
        smape = pd.Series(self.smape(y_true, y_pred))

        mapes_zero = mape_func.evaluate_by_index(y_true, y_true_mean) * 100
        maes_zero = mae_func.evaluate_by_index(y_true, y_true_mean)

        def quan_75(y, quan=0.75):
            return y.quantile(quan)

        def quan_95(y, quan=0.95):
            return y.quantile(quan)

        stats_list = ['mean', 'median', quan_75, quan_95, 'std']

        res = pd.DataFrame([maes.agg(stats_list),
                            maes_zero.agg(stats_list),
                            mapes.agg(stats_list),
                            mapes_zero.agg(stats_list),
                            smape.agg(stats_list)],
                           index=['maes', 'maes_zero', 'mapes', 'mapes_zero', "smapes"])

        # res.loc["r2",:] = r2_score(y_true, y_pred)

        return res

    def plot_diagnostics(data, dt_col, resid_col):
        """
        Строим визуальную оценку остатков - остатки as-is, гистограмма остатков, qqplot и автокореляционную функцию.
        """

        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # resid as-is
        sns.lineplot(data=data, x=dt_col, y=resid_col, ax=ax[0][0]);
        ax[0][0].axhline(0, c="k", linestyle="--")
        ax[0][0].set_title("Динамика ошибок прогноза")
        ax[0][0].set_xlabel(None)
        ax[0][0].set_ylabel(None)
        ax[0][0].xaxis.set_major_locator(plt.MaxNLocator(4))

        # hist
        sns.histplot(data=data, x=resid_col, ax=ax[0][1]);
        ax[0][1].set_title("Гистограмма ошибок прогноза")
        ax[0][1].set_xlabel(None)
        ax[0][1].set_ylabel(None)

        # acf
        sm.graphics.tsa.plot_acf(data[resid_col], lags=10, ax=ax[1][0]);
        ax[1][0].set_title("Автокореляции ошибок прогноза")
        ax[1][0].set_xlabel(None)
        ax[1][0].set_ylabel(None)

        # qq
        qqplot(data[resid_col], line='s', ax=ax[1][1]);
        ax[1][1].set_title("Q-q график ошибок прогноза")
        ax[1][1].set_xlabel(None)
        ax[1][1].set_ylabel(None)

    def test_trend(self, row):
        """Case 1: Both tests conclude that the series is not stationary - The series is not stationary
            Case 2: Both tests conclude that the series is stationary - The series is stationary
            Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
            Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
            """
        kpsstest = kpss(row, regression="c", nlags="auto")
        dftest = adfuller(row, autolag="AIC")
        nonstationary_kpss = kpsstest[1] < 0.05
        nonstationary_df = dftest[1] > 0.05
        return nonstationary_kpss | nonstationary_df, kpsstest[1], dftest[1]

    def prophet_forecaster(self, ts, h_forecast=7):
        """
        Prophet с перебором параметров на кросс-валидации. Использует преобразование бокса-кокса на случай, если
        нету одной ярко выраженной  сезонности.
        """
        ts_train = ts[:-h_forecast].squeeze()
        # здесь для валидации на длинных отрезках используем 5-7 фолдов
        cv = ExpandingWindowSplitter(fh=list(range(1, h_forecast)), initial_window=len(ts_train) - (h_forecast * 2),
                                     step_length=int(h_forecast / 6))
        transformer = BoxCoxTransformer(method="guerrero", sp=7)
        ts_train_box = transformer.fit_transform(ts_train)
        forecaster = prophet_sktime()
        param_grid = {
            "seasonality_mode": ["additive", "multiplicative"],
            "changepoint_prior_scale": [0.05, 0.1, 2.5, 10],
            "seasonality_prior_scale": [1, 5, 15, 20],
            "add_seasonality": [{'name': 'weekly',
                                 'period': 7,
                                 'fourier_order': 5}]
        }
        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            param_grid=param_grid,
            cv=cv,
            verbose=1,
            strategy="refit"
        )  # по умолчанию подбирает по mape, за это отвечает параметр scoring
        gscv.fit(ts_train_box)
        print(gscv.best_params_)
        print(gscv.best_score_)
        forecaster_final = prophet_sktime(**gscv.best_params_)

        forecaster_final.fit(ts_train_box)
        fh_relative = ForecastingHorizon([i for i in range(1, h_forecast + 1)], is_relative=True)
        y_pred = self.BC_inverse(forecaster_final.predict(fh_relative), transformer.lambda_, ts_train_box.std())
        return y_pred

    def prophet_forecaster_fourier(self, ts, h_forecast=7):
        """
        Использует выделение тренда из коробки,  остотчная част рассматривается, как сезонность, моделирутеся фурьями
        """
        ts_train = ts[:-h_forecast].squeeze()
        ts_train = ts_train.reset_index()
        ts_train.columns = ['ds', 'y']

        m = Prophet()
        m.fit(ts_train)
        future = m.make_future_dataframe(periods=h_forecast)

        # Прогнозирование
        forecast = m.predict(future)

        # Извлечение только трендовой компоненты
        trend_forecast = forecast[['ds', 'trend']]
        ts_seasonal = ts_train.y - trend_forecast.trend[:-h_forecast]
        y_pred = self.fourierExtrapolation(ts_seasonal, h_forecast, 800)[-h_forecast:] + trend_forecast.trend[
                                                                                         -h_forecast:]
        y_pred.index = pd.Index(trend_forecast[-h_forecast:].ds)
        return y_pred

    def results_per_model(self, list_df):
        """
        Собирает репорт по метрикам для разных горизнтов прогнозирования, берёт mapes
        и медиану. Строит новый сводный отчёт

        """
        interested_metrics = [('mapes', 'mean'), ('mapes', 'median'), ('smapes', 'mean')]
        columns = []
        for tup in interested_metrics:
            columns.append(f"{tup[0].strip()}_{tup[1].strip()}")
        values_pivot = []
        for df in list_df:
            values_pivot.append(df.stack().loc[interested_metrics].values)
        combined_array = np.vstack(values_pivot)
        return pd.DataFrame(combined_array, columns=columns, index=[5, 30, 90])

    def estimate_envelope(self, forecaster, ts):
        """

        :param forecaster:
        :param ts:
        :return:
        """
        predicts = []
        reports = []
        for h in self.h_forecasts:
            y_pred = forecaster(ts, h)
            report = self.metrics_report(ts[-h:].y.values, y_pred.values)
            predicts.append(y_pred)
            reports.append(report)
        for y_pred, h_forecast in zip(predicts, self.h_forecasts):
            plt.plot(y_pred.index, y_pred.values, label=f"{h_forecast} d ahead")
        plt.plot(ts[-90:].index, ts[-90:].y.values, c="black", label="fact")
        plt.legend()
        plt.show()
        cumulative_report = self.results_per_model(reports)
        print(cumulative_report)

    def fourierExtrapolation(self, x_orig, n_predict, n_harm=20, inner_freq=1.0):
        """
        Функция для экстраполяции рядов Фурье. По вр х делает предикт на n_predict шагов
        с помощью частот с наибольшей амплитудой
        Возвращает исходный ряд + экстраполяцию на n_predict шагов

        https://gist.github.com/tartakynov/83f3cd8f44208a1856ce
        """
        n_orig = x_orig.size
        n = n_orig - len(x_orig) % 7
        t = np.arange(0, n)
        x = x_orig.iloc[:n]
        x_freqdom = fft.fft(x, axis=0)  # x in frequency domain
        f = fft.fftfreq(n, inner_freq)  # frequencies
        indexes = list(range(n))
        indexes.sort(key=lambda i: np.absolute(x_freqdom[i]), reverse=True)

        t = np.arange(0, n_orig + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[:1 + n_harm]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t * inner_freq + phase)
        return restored_sig

    def es_forecaster(self, ts, h_forecast=7):
        ts.index = pd.PeriodIndex(ts.index, freq='D')
        ts_train = ts[:-h_forecast]
        param_grid = {
            "sp": [7, 365, "auto"],
            "seasonal": ["add", "mul"],
            "trend": ["add", "mul"],
            "damped_trend": [True, False],
        }
        cv = ExpandingWindowSplitter(fh=list(range(1, h_forecast)), initial_window=len(ts_train) - (h_forecast * 2),
                                     step_length=int(h_forecast / 6))
        forecaster = ExponentialSmoothing()

        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            param_grid=param_grid,
            cv=cv,
            verbose=1,
            strategy="refit"
        )
        gscv.fit(ts_train)
        forecaster_final = ExponentialSmoothing(**gscv.best_params_)
        forecaster_final.fit(ts_train.squeeze())
        fh_relative = ForecastingHorizon([i for i in range(1, h_forecast + 1)], is_relative=True)
        y_pred = forecaster_final.predict(fh_relative)
        y_pred.index = y_pred.index.to_timestamp()
        ts.index = ts.index.to_timestamp()
        return y_pred

    def stl_arima_forecaster(self, ts, h_forecast=7):
        """
        STL для разбиения на компоненты. Для прогнозирования сезонности и остатков используются
        фурье и автоарима, трендовая компонента прогнозируется наивно
        :param ts:
        :param h_forecast:
        :return: y_pred
        """
        ts.index = pd.PeriodIndex(ts.index, freq='D')
        ts_train = ts[:-h_forecast]
        m = STLForecaster(sp=7, forecaster_trend=NaiveForecaster(strategy="drift"),
                          forecaster_resid=AutoARIMA(suppress_warnings=True))
        fh = list(range(1, h_forecast + 1))
        m.fit(ts_train)
        y_pred = m.forecaster_trend_.predict(fh=fh)
        y_pred += m.forecaster_resid_.predict(fh=fh).values
        y_pred += self.fourierExtrapolation(m.seasonal_, h_forecast, 800)[-h_forecast:]
        y_pred.index = y_pred.index.to_timestamp()
        ts.index = ts.index.to_timestamp()
        return y_pred

    def ets_forecaster(self, ts, h_forecast=7):
        ts.index = pd.PeriodIndex(ts.index, freq='D')
        ts_train = ts[:-h_forecast]
        model = StatsForecastAutoETS()
        model.fit(ts_train)
        y_pred = model.predict(fh=np.arange(1, h_forecast + 1))
        y_pred.index = y_pred.index.to_timestamp()
        ts.index = ts.index.to_timestamp()
        return y_pred

    def baseline_filter(self, ts):
        """
        Базовая обработка временного ряда. Убирает последовательность нулей в самом начале, так как некоторые ряды
        имеют по 200-300 с начала отсчёта. IForest-ом  находит выбросы, заменяет их на медиану соседних. Поднимает
        ряд на 1, чтобы нормально считалось MAPE на кросс-валидации, и в будущем для других метрик
        """
        first_date_null = None
        last_date_null = None
        k = 0
        for date in ts.index:
            if ts.loc[date, "y"] == 0 and first_date_null == None:
                first_date_null = date
                k += 1
            elif ts.loc[date, "y"] == 0:
                k += 1
            elif ts.loc[date, "y"] != 0 and first_date_null != None:
                last_date_null = date
                break
        ts_f = ts.copy()
        if k > 100:
            ts_f = ts.loc[last_date_null:]

        ts_f = ts_f + 1  # поднимем на единицу, чтобы mape нормально считалась везде
        window_size = 3
        rolling_median = ts_f.rolling(window=window_size, min_periods=1).median()
        model = IsolationForest(contamination=0.05)
        outliers = model.fit_predict(ts_f)
        outliers = outliers == -1
        ts_f[outliers] = rolling_median[outliers]
        return ts_f

    def fourier_analysis(self, ts, inner_freq=1):
        fft_values = fft.fft(ts, axis=0)
        N = len(ts)
        freq_values = fft.fftfreq(N, inner_freq)
        indexes = list(range(N))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(fft_values[i]), reverse=True)
        leak_step = max(indexes[1:])
        plt.plot(freq_values[:N // 2], np.abs(fft_values[:N // 2]), label=f"Year")
        plt.legend()
        return round(1 / np.abs(freq_values[leak_step]))

    def primary_analysis(self, ts_original):
        """
        Рисует исходный ряд, предобработанный  baseline_filter
        Приводит спектр фурье по неделям, по годам. Рисует ACF и PCF
        :param ts_original:
        :return:
        """
        fig, ax = plt.subplots(3, 2, figsize=(20, 16))
        ts = self.baseline_filter(ts_original)

        ax[0][0].plot(ts_original.index, ts_original.y, label='Original TS')
        ax[0][0].legend()

        ax[0][1].plot(ts.index, ts.y, label='Preprocessed TS')
        ax[0][1].legend()

        inner_freq = 1 / 365
        fft_values = fft.fft(ts_original.y, axis=0)
        N = len(ts)
        freq_values = fft.fftfreq(N, inner_freq)
        ax[1, 1].plot(freq_values[:N // 2][:10], np.abs(fft_values[:N // 2][:10]), label=f"Year")
        ax[1, 1].set_title("Фурье спектр")
        ax[1, 1].legend()

        inner_freq = 1 / 7
        fft_values = fft.fft(ts_original.y, axis=0)
        N = len(ts)
        freq_values = fft.fftfreq(N, inner_freq)

        ax[1, 0].plot(freq_values[:N // 2], np.abs(fft_values[:N // 2]), label=f"Weeks")
        ax[1, 0].set_title("Фурье спектр")
        ax[1, 0].legend()
        tsaplots.plot_acf(ts.dropna(), ax=ax[2, 0], lags=40);
        tsaplots.plot_pacf(ts.dropna(), ax=ax[2, 1], lags=40);
        plt.show()
        print(self.test_trend(ts))

    def run_pipeline(self, item_name):
        ts = self.get_item_data(item_name)
        self.primary_analysis(ts)
        ts_treated = self.baseline_filter(ts)
        print(f"Item: {item_name}")
        print("--------------Prophet from box--------------")
        self.estimate_envelope(self.prophet_forecaster, ts_treated)
        print("--------------Prophet trend + Fourier--------------")
        self.estimate_envelope(self.prophet_forecaster_fourier, ts_treated)
        print("--------------Stl + Fourier + Arima--------------")
        self.estimate_envelope(self.stl_arima_forecaster, ts_treated)
        """
        print("--------------Exponential Smoothing--------------")
        self.estimate_envelope(self.es_forecaster,ts_treated)
        """
        print("--------------Auto ets--------------")
        self.estimate_envelope(self.ets_forecaster, ts_treated)

    def run_pipeline_boost(self, item_name):
        ts_test = self.get_item_data_boost(item_name)
        filtered = self.baseline_filter(ts_test[["y"]])
        ts_test = ts_test.loc[filtered.index, :]
        ts_test.y = filtered.y
        self.estimate_envelope(self.boost_forecaster, ts_test)

    def prophet_detrender(self, ts, h_forecast):
        """
        Prophet с перебором параметров на кросс-валидации. Использует преобразование бокса-кокса на случай, если
        нету одной ярко выраженной  сезонности.
        """
        ts_train = ts.squeeze()
        # здесь для валидации на длинных отрезках используем 5-7 фолдов
        cv = ExpandingWindowSplitter(fh=list(range(1, h_forecast)), initial_window=len(ts_train) - (h_forecast * 2),
                                     step_length=int(h_forecast / 6))
        forecaster = prophet_sktime()
        param_grid = {
            "seasonality_mode": ["multiplicative", "additive"],
        }
        gscv = ForecastingGridSearchCV(
            forecaster=forecaster,
            param_grid=param_grid,
            cv=cv,
            verbose=-1,
            strategy="refit"
        )
        gscv.fit(ts_train)

        ts_train = ts_train.reset_index()
        ts_train.columns = ['ds', 'y']

        m = Prophet(**gscv.best_params_)
        m.fit(ts_train)
        future = m.make_future_dataframe(periods=h_forecast)

        # Прогнозирование
        forecast = m.predict(future)
        # Извлечение только трендовой компоненты
        trend_forecast = forecast[['ds', 'trend']]
        if gscv.best_params_['seasonality_mode'] == 'multiplicative':
            ts_seasonal = ts_train.y / trend_forecast.trend[:-h_forecast]
        else:
            ts_seasonal = ts_train.y - trend_forecast.trend[:-h_forecast]
        return ts_seasonal, trend_forecast.trend[-h_forecast:], gscv.best_params_['seasonality_mode']

    def boost_forecaster(self, ts, h_forecast):
        x_train, y_train, x_test, y_test = self.split_train_test(ts, h_forecast)
        y_train, trend_forecast, seasonal_mode = self.prophet_detrender(y_train, h_forecast)
        m = self.fit_lightgbm(x_train, y_train)
        forecast = m.predict(x_test)
        if seasonal_mode == "multiplicative":
            y_pred = forecast * trend_forecast.values
        else:
            y_pred = forecast + trend_forecast.values

        return pd.Series(y_pred, index=y_test.index)

    def split_train_test(self, df, h_forecast=7, train_ratio=None):
        split_t = h_forecast

        y = df['y']
        y_train = y[:-split_t]
        y_test = y[-split_t:]

        xdf = df.drop('y', inplace=False, axis=1)
        x_train = xdf[:-split_t]
        x_test = xdf[-split_t:]

        return x_train, y_train, x_test, y_test

    def fit_lightgbm(self, x_train, y_train, n_estimators=100, verbose_eval=50):
        model = lightgbm.LGBMRegressor(
            boosting_type='gbdt',
            n_estimators=n_estimators,
            verbose=-1)

        model.fit(x_train,
                  y_train,
                  eval_set=[(x_train, y_train)],
                  eval_metric='mape',
                  )

        return model
