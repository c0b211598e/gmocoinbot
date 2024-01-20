import glob
import json
import logging
import os
import pickle
import time
import traceback
import warnings
from datetime import date, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from logging import DEBUG, StreamHandler, getLogger

import joblib
import numpy as np
import pandas as pd
import requests
import talib

from config import GMOBotConfig
from gmocoin import GMOCoin
from richman_features import calc_features

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)

formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(funcName)s,%(lineno)d] %(message)s"
)

handler.setFormatter(formatter)
logger.propagate = False

feature_pkl_path = GMOBotConfig.feature_pkl_path
symbol = GMOBotConfig.symbol
pips = GMOBotConfig.pips
gmo_api = GMOCoin()

warnings.filterwarnings("ignore")


def get_file_list(input_dir):
    return sorted(
        [
            p
            for p in glob.glob(os.path.join(input_dir, "**"), recursive=True)
            if os.path.isfile(p)
        ]
    )


def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def load_pickle(load_path):
    with open(load_path, mode="rb") as f:
        return pickle.load(f)


def save_pickle(object, save_path):
    with open(save_path, mode="wb") as f:
        pickle.dump(object, f)


def round_off_to_two_decimal_places(input_val: float) -> float:
    TWOPLACES = Decimal(10) ** -2
    output = Decimal(str(input_val)).quantize(TWOPLACES, rounding=ROUND_DOWN)
    return float(output)


def get_gmo_ohlcv(day, interval):
    """
    ある特定の日の特定の足のOHLCVデータを取得
    """
    endPoint = "https://api.coin.z.com/public"
    path = "/v1/klines?symbol=BTC_JPY&interval=" + interval + "min&date=" + day

    response = requests.get(endPoint + path)
    cnt = 0

    if "data" not in response.json():
        logger.info(json.dumps(response.json(), indent=2))
    while "data" not in response.json():
        time.sleep(1.1)
        response = requests.get(endPoint + path)
        cnt += 1
        if cnt == 10:
            raise TimeoutError
    df = pd.DataFrame(response.json()["data"])
    df.rename(
        columns={
            "openTime": "timestamp",
            "open": "op",
            "high": "hi",
            "low": "lo",
            "close": "cl",
        },
        inplace=True,
    )
    df["op"] = df["op"].astype(float)
    df["hi"] = df["hi"].astype(float)
    df["lo"] = df["lo"].astype(float)
    df["cl"] = df["cl"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def get_latest_ohlcv(interval, last_day):
    """
    直近のOHLCVデータをすべて取得
    """
    today = datetime.today()
    start = time.time()
    for i in range(last_day, 0, -1):
        day = today - timedelta(days=i)
        day = day.strftime("%Y%m%d")
        if i == last_day:
            df_tmp = get_gmo_ohlcv(day, str(interval))
            time.sleep(1.1)
        else:
            df_tmp = pd.concat((df_tmp, get_gmo_ohlcv(day, str(interval))))
            if i > 1:
                time.sleep(1.1)
    dt_now = datetime.now()
    if dt_now.hour >= 0 and dt_now.hour < 6:
        df = df_tmp
    else:
        today = datetime.today()
        today = today.strftime("%Y%m%d")
        df = pd.concat((df_tmp, get_gmo_ohlcv(today, str(interval))))
    return df


class GMOBot:
    """
    ボットのクラスです。各ボットごとに注文処理を行い、注文ID、ポジションIDを管理します。
    """

    def __init__(
        self,
        model_buy_path,
        model_sell_path,
        atr_coeff,
        orderId_pkl_path,
        positionId_pkl_path,
        lot=0.01,
        atr_period=14,
    ):
        self.positionId_pkl_path = positionId_pkl_path
        self.orderId_pkl_path = orderId_pkl_path
        self.atr_coeff = atr_coeff
        if os.path.isfile(self.positionId_pkl_path):
            self.open_positionId_dict = load_pickle(self.positionId_pkl_path)
        else:
            self.open_positionId_dict = {}
        if os.path.isfile(self.orderId_pkl_path):
            self.orderId_dict = load_pickle(self.orderId_pkl_path)
        else:
            self.orderId_dict = {}
        self.order_lot = lot
        self.model_buy = joblib.load(model_buy_path)
        self.model_sell = joblib.load(model_sell_path)
        self.features = load_pickle(feature_pkl_path)
        self.atr_period = atr_period

    def update_open_positionIds(self, df_executions):
        executed_open_orders = set()
        executed_position = set()
        for orderId, positionId, settleType, side, size, price in zip(
            df_executions.orderId.values.tolist(),
            df_executions.positionId.values.tolist(),
            df_executions.settleType.values.tolist(),
            df_executions.side.values.tolist(),
            df_executions.size_.values.tolist(),
            df_executions.price.values.tolist(),
        ):

            if orderId in self.orderId_dict and settleType == "OPEN":
                self.open_positionId_dict[positionId] = (
                    settleType,
                    side,
                    round(float(size), 2),
                    int(price),
                )
                executed_open_orders.add(orderId)
            if (
                positionId in self.open_positionId_dict
                and self.open_positionId_dict[positionId][0] == "OPEN"
                and settleType == "CLOSE"
            ):

                executed_position.add(positionId)

        for positionId in list(executed_position):
            del self.open_positionId_dict[positionId]

        save_pickle(self.open_positionId_dict, self.positionId_pkl_path)
        save_pickle(self.orderId_dict, self.orderId_pkl_path)

    def cancel_orders(self):
        cancel_orderId_list = list(self.orderId_dict.keys())
        if cancel_orderId_list:
            gmo_api.create_cancel_multiple_orders(cancel_orderId_list)
            time.sleep(0.5)
            for orderId in cancel_orderId_list:
                del self.orderId_dict[orderId]
        save_pickle(self.orderId_dict, self.orderId_pkl_path)

    def create_limit(self, side, size, price):
        if size > 0 and size < 0.1:
            res = gmo_api.create_limit_order(symbol, side, size, price)
            orderId = int(res["data"])
            return orderId
        else:
            logger.info("order size is invalid : 0 < size < 0.1")
            return 0

    def close_limit(self, side, size, price, positionId):
        res = gmo_api.create_limit_close_order(symbol, side, size, price, positionId)
        orderId = int(res["data"])
        return orderId

    def close_market(self, side, size, positionId):
        gmo_api.create_market_close_order(symbol, side, size, positionId)
        return None

    def get_latest_order_price(self, df):
        hi, lo, cl = df["hi"].values, df["lo"].values, df["cl"].values
        df["ATR"] = talib.ATR(hi, lo, cl, timeperiod=self.atr_period)
        limit_price_dist = df["ATR"] * self.atr_coeff
        limit_price_dist = (
            np.maximum(1, (limit_price_dist / pips).round().fillna(1)) * pips
        )
        df["buy_price"] = df["cl"] - limit_price_dist
        df["sell_price"] = df["cl"] + limit_price_dist

        buy_price = int(df["buy_price"].iloc[-2])
        sell_price = int(df["sell_price"].iloc[-2])

        return buy_price, sell_price

    def predict_order(self, df_features):
        df_features = df_features.copy()[-5:]
        df_features["y_predict_buy"] = self.model_buy.predict(
            df_features[self.features]
        )
        df_features["y_predict_sell"] = self.model_sell.predict(
            df_features[self.features]
        )

        predict_buy = df_features["y_predict_buy"].iloc[-2]
        predict_sell = df_features["y_predict_sell"].iloc[-2]
        logger.info(f"predict_buy : {predict_buy} predict_sell : {predict_sell}")
        return predict_buy, predict_sell

    def get_buysell_signals(self, df_features):
        predict_buy, predict_sell = self.predict_order(df_features)
        buy_signal = predict_buy > 0
        sell_signal = predict_sell > 0
        return buy_signal, sell_signal

    def entry_position(
        self,
        df_features,
        buy_price,
        sell_price,
    ):
        buy_signal, sell_signal = self.get_buysell_signals(df_features)
        buy_size = 0.0
        sell_size = 0.0
        for _, (_, side, size, _) in self.open_positionId_dict.items():
            if side == "BUY":
                buy_size += size
            if side == "SELL":
                sell_size += size

        if buy_signal and buy_size == 0.0:
            args = ("BUY", self.order_lot, buy_price)
            orderId = self.create_limit(*args)
            if orderId != 0:
                self.orderId_dict[orderId] = args
            logger.info(
                f"entry_buy_order : side : {args[0]} size : {args[1]} price : {args[2]}"
            )
        if sell_signal and sell_size == 0.0:
            args = ("SELL", self.order_lot, sell_price)
            orderId = self.create_limit(*args)
            if orderId != 0:
                self.orderId_dict[orderId] = args
            logger.info(
                f"entry_sell_order : side : {args[0]} size : {args[1]} price : {args[2]}"
            )
        return None

    def exit_position(self, buy_price, sell_price):
        if self.open_positionId_dict != {}:
            for open_positionId, (
                _,
                side,
                size,
                _,
            ) in self.open_positionId_dict.items():
                if side == "BUY":
                    args = ("CLOSE-BUY", size, sell_price)
                    orderId = self.close_limit(
                        "SELL", size, sell_price, open_positionId
                    )
                    self.orderId_dict[orderId] = args
                    logger.info(
                        f"exit_buy_position => sell order : size : {size} price : {sell_price}"
                    )
                else:
                    args = ("CLOSE-SELL", size, buy_price)
                    orderId = self.close_limit("BUY", size, buy_price, open_positionId)
                    self.orderId_dict[orderId] = args
                    logger.info(
                        f"exit_sell_position => buy order : size : {size} price : {buy_price}"
                    )

        return None

    def exit_position_market(self):
        if self.open_positionId_dict != {}:
            for open_positionId, (
                _,
                side,
                size,
                _,
            ) in self.open_positionId_dict.items():
                if side == "BUY":
                    self.close_market("SELL", size, open_positionId)
                    logger.info(f"exit_buy_position => sell order : size : {size}")
                else:
                    self.close_market("BUY", size, open_positionId)
                    logger.info(f"exit_sell_position => buy order : size : {size}")

        return None

    def exit_and_entry(self, df_features):
        buy_price, sell_price = self.get_latest_order_price(df_features)
        self.exit_position(buy_price, sell_price)
        self.entry_position(df_features, buy_price, sell_price)


class Manager:
    """
    複数のボットを管理するクラスです。ボットごとにロットを自動的に割り当て、各ボットの注文管理をします。
    """

    def __init__(self, model_buy_dir, model_sell_dir, available_margin, max_lot=0.01):
        logger.info("Initialization start...")
        self.available_margin = available_margin
        assert self.available_margin < 0.85 and self.available_margin >= 0.0
        model_buy_list = get_file_list(model_buy_dir)
        model_sell_list = get_file_list(model_sell_dir)
        max_lot *= 100
        if len(model_buy_list) == len(model_sell_list):
            number_of_models = len(model_buy_list)
            lot_list = [
                (max_lot + i) // number_of_models / 100 for i in range(number_of_models)
            ][::-1]
            logger.info(
                f"number of total bots is {number_of_models} and lot list (place holder) is {lot_list}"
            )

            self.bot_dict = {}
            for model_buy_path, model_sell_path, lot in zip(
                model_buy_list, model_sell_list, lot_list
            ):
                bot_id = model_buy_path.split("_")[-1].split(".")[0]
                atr_coeff = float(
                    model_buy_path.split("_")[-1].split(".")[0].replace("p", ".")
                )
                orderId_pkl_path = f"/work/cache/bot_{bot_id}_orderId_list.pkl"
                positionId_pkl_path = f"/work/cache/bot_{bot_id}_positionId_list.pkl"

                logger.info(f"creating bot_{bot_id}...")
                self.bot_dict[f"bot_{bot_id}"] = GMOBot(
                    model_buy_path=model_buy_path,
                    model_sell_path=model_sell_path,
                    atr_coeff=atr_coeff,
                    lot=lot,
                    orderId_pkl_path=orderId_pkl_path,
                    positionId_pkl_path=positionId_pkl_path,
                )
            logger.info(
                f"{number_of_models} bots are successfully generated and all the position status are initialized."
            )

        else:
            logger.info(
                "The number of model_buy_list and model_sell_list is not identincal."
            )

    def update_order_lot_all_bots(self, df):
        number_of_models = len(self.bot_dict.keys())
        remaining_JPY = gmo_api.get_position_assets()
        max_lot = round_off_to_two_decimal_places(
            remaining_JPY / df["cl"][-1] * self.available_margin
        )
        max_lot *= 100

        lot_list = [
            (max_lot + i) // number_of_models / 100 for i in range(number_of_models)
        ][::-1]

        for bot_name, new_lot in zip(self.bot_dict.keys(), lot_list):
            self.bot_dict[bot_name].order_lot = new_lot

        logger.info(f"max lot is updated. new max lot is {max_lot/100}")
        logger.info(f"new lot_list is {lot_list}")

    def update_status_of_all_bots(self):
        df_executions = gmo_api.get_latest_execution_df()
        for bot_name in self.bot_dict.keys():
            self.bot_dict[bot_name].update_open_positionIds(df_executions)

    def update_ohlcv(self):
        df = pd.read_pickle("/work/data/ohlcv_gmo_15m.pkl")
        delta_day = (datetime.now() - df.index[-1]).days
        df_latest = get_latest_ohlcv(15, delta_day + 1)
        df = pd.concat([df, df_latest[~df_latest.index.isin(df.index)]]).sort_index()
        df[:-1].to_pickle("/work/data/ohlcv_gmo_15m.pkl")
        return df

    def calculate_features(self, df):
        df_copy = df.copy()
        df_sliced = df_copy[df_copy.index > pd.to_datetime("2021-08-01 00:00")]
        df_features = calc_features(df_sliced)
        return df_features

    def exit_and_entry_all_bots(self, df_features):
        for bot_name in self.bot_dict.keys():
            logger.info(f"{bot_name} is starting orders.")
            self.bot_dict[bot_name].exit_and_entry(df_features)

    def close_position_market_all_bots(self):
        for bot_name in self.bot_dict.keys():
            self.bot_dict[bot_name].exit_position_market()

    def cancel_multiple_orders_all_bots(self):
        for bot_name in self.bot_dict.keys():
            self.update_status_of_all_bots()
            self.bot_dict[bot_name].cancel_orders()
        time.sleep(1.0)

    def cancel_all_orders_all_bots(self):
        self.update_status_of_all_bots()
        try:
            gmo_api.create_cancel_all_order("BTC_JPY")
            while True:
                orders = gmo_api.get_active_orders("BTC_JPY")
                time.sleep(1.5)
                gmo_api.create_cancel_all_order("BTC_JPY")
                time.sleep(1.5)
                if len(orders["data"]) == 0:
                    break
            self.update_status_of_all_bots()
            for bot_name in self.bot_dict.keys():
                self.bot_dict[bot_name].orderId_dict = {}

        except Exception as e:
            logger.error(traceback.format_exc())
            pass

    def get_open_positionId_all_bots(self):
        self.update_status_of_all_bots()
        all_open_positionId_dict = {}
        for bot_name in self.bot_dict.keys():
            logger.info(
                f"{bot_name} : open positonId_dict is {self.bot_dict[bot_name].open_positionId_dict}"
            )
            all_open_positionId_dict.update(
                self.bot_dict[bot_name].open_positionId_dict
            )
        logger.info(f"positionId_list : {all_open_positionId_dict}")

        return all_open_positionId_dict

    def get_orderId_all_bots(self):
        for bot_name in self.bot_dict.keys():
            logger.info(
                f"{bot_name} : orderId_dict is {self.bot_dict[bot_name].orderId_dict}"
            )

    def get_all_open_positionIds(self, df_executions):
        all_open_positionId_dict = {}
        for orderId, positionId, settleType, side, size, price in zip(
            df_executions.orderId.values.tolist()[::-1],
            df_executions.positionId.values.tolist()[::-1],
            df_executions.settleType.values.tolist()[::-1],
            df_executions.side.values.tolist()[::-1],
            df_executions.size_.values.tolist()[::-1],
            df_executions.price.values.tolist()[::-1],
        ):

            if settleType == "OPEN":
                all_open_positionId_dict[positionId] = (
                    settleType,
                    side,
                    round(float(size), 2),
                    int(price),
                )
            if positionId in all_open_positionId_dict and settleType == "CLOSE":
                del all_open_positionId_dict[positionId]

        logger.info(f"positionId_list_from_execution : {all_open_positionId_dict}")

        return all_open_positionId_dict

    def close_market(self, side, size, positionId):
        gmo_api.create_market_close_order(symbol, side, size, positionId)
        return None

    def detect_and_close_untracked_positionId(self):
        df_executions = gmo_api.get_latest_execution_df()
        position_dict_from_exectuion = self.get_all_open_positionIds(df_executions)
        tracked_positon_dict = self.get_open_positionId_all_bots()

        untracked_positionId_dict = position_dict_from_exectuion
        for key in tracked_positon_dict.keys():
            del untracked_positionId_dict[key]

        if not untracked_positionId_dict:
            logger.info(f"untracked_positionId_dict : {untracked_positionId_dict}")

        for open_positionId, (
            _,
            side,
            size,
            _,
        ) in untracked_positionId_dict.items():
            if side == "BUY":
                self.close_market("SELL", size, open_positionId)
                logger.info(
                    f"exit_untracked_buy_position => sell order : size : {size}"
                )
            else:
                self.close_market("BUY", size, open_positionId)
                logger.info(
                    f"exit_untracked_sell_position => buy order : size : {size}"
                )

    def detect_and_close_untracked_allposition(self):
        tracked_positon_dict = self.get_open_positionId_all_bots()
        position = gmo_api.get_position(symbol)
        logger.info(
            f"tracked_positon_dict: {tracked_positon_dict} position: {position}"
        )
        if not tracked_positon_dict and position["buy"] > 0:
            gmo_api.create_market_close_bulk_order(symbol, "SELL", position["buy"])
            logger.info(
                f"exit_untracked_buy_position_all => sell order : size : {position['buy']}"
            )
        if not tracked_positon_dict and position["sell"] > 0:
            gmo_api.create_market_close_bulk_order(symbol, "BUY", position["sell"])
            logger.info(
                f"exit_untracked_sell_position_all => sell order : size : {position['sell']}"
            )

    def start_all_bots(self):

        self.update_status_of_all_bots()
        self.cancel_multiple_orders_all_bots()
        time.sleep(1.0)
        self.cancel_all_orders_all_bots()
        logger.info("all the status of all bots are updated.")
        self.get_open_positionId_all_bots()
        df = self.update_ohlcv()
        df_features = self.calculate_features(df)
        self.update_order_lot_all_bots(df)

        current_min = datetime.now().minute
        prev = (current_min // 15) * 15

        while True:
            dt_now = datetime.now()
            pre_maintainance = (
                date.today().weekday() == 2 and dt_now.hour == 14 and dt_now.minute == 1
            )
            maintainance = (
                date.today().weekday() == 2 and dt_now.hour >= 14 and dt_now.hour <= 17
            )
            # メンテナンス時はポジションをもたないようにする
            if pre_maintainance:
                try:
                    self.update_status_of_all_bots()
                    self.cancel_multiple_orders_all_bots()
                    self.cancel_all_orders_all_bots()
                    self.close_position_market_all_bots()
                    time.sleep(20)
                    self.get_open_positionId_all_bots()
                    self.get_orderId_all_bots()
                    time.sleep(60)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    pass

            # メンテナンス以外のときは取引をする
            if not maintainance:
                try:
                    time.sleep(5.0)
                    self.update_status_of_all_bots()
                    df = self.update_ohlcv()
                    if df.index[-1].minute == dt_now.minute and dt_now.minute != prev:
                        prev = dt_now.minute
                        self.cancel_multiple_orders_all_bots()
                        self.cancel_all_orders_all_bots()
                        self.update_order_lot_all_bots(df)
                        df = self.update_ohlcv()
                        df_features = self.calculate_features(df)
                        self.detect_and_close_untracked_positionId()
                        self.detect_and_close_untracked_allposition()
                        self.exit_and_entry_all_bots(df_features)
                        self.get_orderId_all_bots()
                        time.sleep(70)
                except Exception as e:
                    logger.error(traceback.format_exc())
                    pass


if __name__ == "__main__":

    model_buy_dir = "/work/model_buy"
    model_sell_dir = "/work/model_sell"
    available_margin = GMOBotConfig.available_margin

    check_create_dir("/work/cache")

    system = Manager(model_buy_dir, model_sell_dir, available_margin)
    system.start_all_bots()
