from sktime.forecasting.fbprophet import Prophet
from sktime.performance_metrics.forecasting import MeanAbsoluteError,MeanAbsolutePercentageError
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

class ProphetLightGMBPredictor():
    def __init__(self, sales, sales_dates, sales_prices, store_id, models_path):
        self.models_path = models_path
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

    def get_preprocessed_item_data(self, item_name):
        ts_item = self.get_item_data(item_name)
        filtered = self.baseline_filter(ts_item[["y"]])
        ts_item = ts_item.loc[filtered.index, :]
        ts_item.y = filtered.y
        return ts_item

    def fit(self, x_train, y_train):
        y_train_seasonal, model_prophet = self.prophet_detrender(y_train)
        model_lightgbm = self.fit_lightgbm(x_train, y_train_seasonal)
        return model_prophet, model_lightgbm

    def predict(self, x_test, model_prophet, model_lightgbm):
        boost_forecast = model_lightgbm.predict(x_test)
        h_forecast = len(x_test)
        future = model_prophet.make_future_dataframe(periods=h_forecast)
        forecast = model_prophet.predict(future)
        trend_forecast = forecast[['ds', 'trend']][-h_forecast:]
        if model_prophet.seasonality_mode == "multiplicative":
            y_pred = boost_forecast * trend_forecast.trend.values
        else:
            y_pred = boost_forecast + trend_forecast.trend.values

        return y_pred

    def prophet_detrender(self, ts):
        """
        Prophet с перебором параметров на кросс-валидации. Использует преобразование бокса-кокса на случай, если
        нету одной ярко выраженной  сезонности.
        """
        ts_train = ts.squeeze()
        ts_train = ts_train.reset_index()
        ts_train.columns = ['ds', 'y']

        seasonality_modes = {'additive': 0, 'multiplicative': 0}
        for mode in seasonality_modes.keys():
            m = Prophet(seasonality_mode=mode)
            m.fit(ts_train)
            y_pred = m.predict(ts_train).yhat.values
            seasonality_modes[mode] = np.mean(self.smape(ts_train.y.values, y_pred))

        best_mode, _ = min(seasonality_modes.items(), key=lambda x: x[1])
        m = Prophet(seasonality_mode=best_mode)
        m.fit(ts_train)

        forecast = m.predict(ts_train)
        # Извлечение только трендовой компоненты
        trend_component = forecast[['ds', 'trend']]
        if best_mode == 'multiplicative':
            ts_seasonal = ts_train.y / trend_component.trend.values
        else:
            ts_seasonal = ts_train.y - trend_component.trend.values
        return ts_seasonal, m

    def split_train_test(self, df, h_forecast=7, train_ratio=None):
        split_t = h_forecast

        y = df['y']
        y_train = y[:-split_t]
        y_test = y[-split_t:]

        xdf = df.drop('y', inplace=False, axis=1)
        x_train = xdf[:-split_t]
        x_test = xdf[-split_t:]

        return x_train, y_train, x_test, y_test

    def save_models(self, item_name, h_forecast, prophet_model, lightgbm_model):
        prophet_model_name = f"prophet_{item_name}_{h_forecast}.pkl"
        lightgbm_model_name = f"lightgbm_{item_name}_{h_forecast}.pkl"
        prophet_model_path = os.path.join(self.models_path, prophet_model_name)
        lightgbm_model_path = os.path.join(self.models_path, lightgbm_model_name)
        print(lightgbm_model_path)
        print(prophet_model_path)
        joblib.dump(prophet_model, prophet_model_path)
        joblib.dump(lightgbm_model, lightgbm_model_path)  # Сохраняем в файл

    def load_models(self, item_name, h_forecast):
        prophet_model_name = f"prophet_{item_name}_{h_forecast}.pkl"
        lightgbm_model_name = f"lightgbm_{item_name}_{h_forecast}.pkl"
        prophet_model_path = os.path.join(self.models_path, prophet_model_name)
        lightgbm_model_path = os.path.join(self.models_path, lightgbm_model_name)
        prophet_model = joblib.load(prophet_model_path)
        lightgbm_model = joblib.load(lightgbm_model_path)
        return prophet_model, lightgbm_model

    def show_result(self, y_pred, y_test):
        plt.plot(y_pred.index, y_pred.values, label="forecast")
        plt.plot(y_test.index, y_test.values, label="fact")
        plt.legend()
        plt.show()
        print(self.metrics_report(y_test, y_pred))

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
