# %%
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from crypto_metrics import CryptoDataTransformation
from crypto_price import CryptoDataExtractor
from decouple import config
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class CryptoPredictor:
    def __init__(self, time_interval="1d", time_hours=24 * 30 * 12 * 4) -> None:
        self.time_interval = time_interval
        self.time_hours = time_hours
        self.save_path = "./datasets"
        self.criptos = ["BTCUSDT"]
        pass

    def download_crypto_dataset(self,):
        API_KEY = config("API_KEY")
        API_SECRET = config("API_SECRET")
        extractor = CryptoDataExtractor(save_path=self.save_path, criptos=self.criptos)
        extractor.from_binance(
            api_key=API_KEY,
            api_secret=API_SECRET,
            time_interval=self.time_interval,
            time_in_hours=self.time_hours,
        )

    def calculate_metrics(self):
        transformer = CryptoDataTransformation(
            save_path="{}/{}".format(self.save_path, self.time_interval),
            criptos=self.criptos,
        )
        transformer.readDataset()

    def load_dataset(self, path="datasets/1d/BTCUSDT.csv"):
        dataset = pd.read_csv(path, sep="|")
        dataset = dataset.set_index("Index")
        del dataset["Date"]
        self.dataset = dataset

    def lag_variables(self, n_lags=10):
        dataset = self.dataset.values
        new_dataset = [
            dataset[:-n_lags],
        ]
        for i in range(1, n_lags + 1):
            index = n_lags - i
            if index == 0:
                new_dataset.append(dataset[i:].copy())
            else:
                new_dataset.append(dataset[i:-index].copy())

        # steps = range(1, n_lags + 1)
        # dataset_lagged = self.dataset.assign(
        #     **{
        #         f"{col} (t-{step})": self.dataset[col].shift(step)
        #         for step in steps
        #         for col in self.dataset
        #     }
        # )
        # dataset_lagged = dataset_lagged.dropna()
        self.dataset_lagged = np.array(new_dataset)
        self.n_lags = n_lags
        # print(dataset_lagged.shape)

    def calculate_labels(self, n_times_future=5):
        dataset = self.dataset_lagged
        self.n_times_future = n_times_future
        last_price = dataset[-1]
        labels = []
        for i in range(1, n_times_future + 1):
            index = n_times_future - i
            if index == 0:
                labels.append(last_price[i:, 3].copy())
            else:
                labels.append(last_price[i:-index, 3].copy())
        self.prediction_labels = np.array(labels)
        # prediction = pd.DataFrame()
        # prediction["Close1"] = self.dataset_lagged[f"Close (t-{self.n_lags})"].shift(-1)
        # prediction["Index"] = self.dataset_lagged.index.values
        # prediction = prediction.set_index("Index")
        # for i in range(2, n_times_future + 1):
        #     prediction["Close" + str(i)] = self.dataset_lagged[f"Close (t-{self.n_lags-i+1})"].shift(-i)
        # self.prediction_labels = prediction

    def normalize_split_dataset(self, split_data=30):
        data_scaler = MinMaxScaler()
        split = split_data
        X = np.hstack(self.dataset_lagged)
        x_train = X[:-split, :]
        x_test = X[-split:, :]
        x_train_n = data_scaler.fit_transform(x_train)
        x_test_n = data_scaler.transform(x_test)
        x_train_n = np.reshape(
            x_train_n,
            (len(x_train), self.dataset_lagged.shape[0], self.dataset_lagged.shape[2]),
        )
        x_test_n = np.reshape(
            x_test_n,
            (len(x_test), self.dataset_lagged.shape[0], self.dataset_lagged.shape[2]),
        )

        close_scaler = MinMaxScaler()
        Y = np.stack(self.prediction_labels, axis=1)

        Y = close_scaler.fit_transform(Y)

        y_train = Y[: (-split + self.n_times_future)]
        y_test = Y[(-split + self.n_times_future) :]

        self.data_scaler = data_scaler
        self.close_scaler = close_scaler
        return x_train_n, x_test_n, y_train, y_test

    def shift_row_prices(self, prices: pd.DataFrame, closes_prices, lastday, n_steps=4):
        dataframes = []
        index = 0
        prices.insert(0, "Close", closes_prices.values, True)
        new_index = prices.index.values
        for i in range(n_steps):
            if self.time_interval == "1d":
                lastday = lastday + datetime.timedelta(days=1)
                new_index = np.append(new_index, lastday.strftime("%m/%d"))
            if self.time_interval == "4h":
                lastday = lastday + datetime.timedelta(hours=4)
                new_index = np.append(new_index, lastday.strftime("%d/%H"))
            if self.time_interval == "1h":
                lastday = lastday + datetime.timedelta(hours=1)
                new_index = np.append(new_index, lastday.strftime("%d/%H"))
        for _, row in prices.iloc[:, :n_steps].iterrows():
            prices_indexs = new_index[index : index + n_steps]
            aux = pd.DataFrame({"price_prediction": row.values}, index=prices_indexs)
            dataframes.append(aux)
            index += 1
        return dataframes

    def draw_prediction(self, x_test_n, n_steps=30, n_future_steps=2):
        close_scaler = self.close_scaler
        prediction = self.regressor.predict(x_test_n[-n_steps:])
        prediction_n = close_scaler.inverse_transform(prediction)
        data_draw = self.dataset[-n_steps:].copy()
        data_draw.index = pd.to_datetime(data_draw.index)
        lastday = data_draw.index[-1]
        if self.time_interval == "1d":
            data_draw.index = data_draw.index.strftime("%m/%d")
        if self.time_interval == "4h":
            data_draw.index = data_draw.index.strftime("%d/%H")
        if self.time_interval == "1h":
            data_draw.index = data_draw.index.strftime("%d/%H")

        plt.figure(figsize=(20, 5))
        plt.plot(data_draw.index, data_draw["Close"], "black")
        plt.scatter(data_draw.index, data_draw["Close"], s=150, alpha=0.3)

        predictions_pd = pd.DataFrame(prediction_n, index=data_draw.index)
        predictions_lines = self.shift_row_prices(
            predictions_pd, data_draw["Close"], lastday, n_steps=n_future_steps
        )

        for line in predictions_lines:
            plt.plot(
                line.index, line.values, color="green", linestyle="--", linewidth=2
            )
            # plt.arrow(line.index[0],line.values[0][0],3,line.values[-1][0]-line.values[0][0], head_width = 0.8, head_length=0.8)

        plt.title("BTC Price, Prediction", fontsize=20)
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("BTC Price(USD)", fontsize=20)
        # plt.legend(loc='best')
        plt.show()

    def set_keras_model(self, keras_model=None):
        if keras_model is None:
            loss_function = "mean_absolute_percentage_error"

            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
            regressor = Sequential()
            regressor.add(
                LSTM(
                    units=25,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(1, 120),
                )
            )
            regressor.add(Dropout(0.5))
            regressor.add(Dense(10, activation="relu"))
            regressor.add(Dropout(0.5))
            regressor.add(Dense(3, activation="relu"))
            regressor.compile(optimizer=optimizer, loss=loss_function, metrics=["mae"])
            self.regressor = regressor
        else:
            self.regressor = keras_model

    def train_keras_model(self, x_train_n, y_train, x_test_n, y_test, show_loss=False):
        batch_size = 32
        num_epochs = 100
        filepath = "./keras_model.h5"
        checkpoint = ModelCheckpoint(
            filepath, monitor="loss", verbose=0, save_best_only=True, mode="min"
        )
        callbacks_list = [checkpoint]

        history = self.regressor.fit(
            x_train_n,
            y_train,
            verbose=1,
            batch_size=batch_size,
            epochs=num_epochs,
            callbacks=callbacks_list,
            validation_data=(x_test_n[: -self.n_times_future], y_test),
            shuffle=False,
        )
        if show_loss is True:
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()
