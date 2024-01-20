"""
### GMOコインの公式APIを使った注文処理 ###

GMOCoinクラスの各インスタンスメソッドは、https://api.coin.z.com/docs/#outline　を参考に作成しました。

サーバー間の時間のずれで"ERR-5008"や"ERR-5009"などのエラーが出て注文処理が通らないときに意図的に時間をずらして注文を通すようにしています。
"""


import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from logging import DEBUG, StreamHandler, getLogger

import pandas as pd
import requests

from config import GMOBotConfig

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

apiKey = GMOBotConfig.apiKey
secretKey = GMOBotConfig.secretKey
td = 5  # サーバー間のズレがあった場合にずらす時間：デフォルト5sec


class GMOCoin:
    def create_limit_order(self, symbol, side, size, price):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/order"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "LIMIT",
            "size": str(round(size, 2)),
            "price": str(int(price)),
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()

        return res.json()

    def create_limit_close_order(self, symbol, side, size, price, positionId):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/closeOrder"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "LIMIT",
            "price": str(int(price)),
            "settlePosition": [{"positionId": positionId, "size": str(round(size, 2))}],
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()

        return res.json()

    def create_market_close_order(self, symbol, side, size, positionId):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/closeOrder"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "settlePosition": [{"positionId": positionId, "size": str(round(size, 2))}],
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()

        return res.json()

    def create_limit_close_bulk_order(self, symbol, side, size, price):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/closeBulkOrder"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "LIMIT",
            "size": str(round(size, 2)),
            "price": str(int(price)),
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()
        return res.json()

    def create_market_order(self, symbol, side, size):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/order"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "size": str(round(size, 2)),
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))
        return res.json()

    def create_market_close_bulk_order(self, symbol, side, size):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/closeBulkOrder"
        reqBody = {
            "symbol": symbol,
            "side": side,
            "executionType": "MARKET",
            "size": str(round(size, 2)),
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()

        return res.json()

    def create_cancel_all_order(self, symbol):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/cancelBulkOrder"
        reqBody = {"symbols": [symbol]}

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()
        return res.json()

    def create_cancel_multiple_orders(self, orders):
        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "POST"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/cancelOrders"
        reqBody = {"orderIds": orders}

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        tmp = res.json()
        while tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5008":
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            text = timestamp + method + path + json.dumps(reqBody)
            sign = hmac.new(
                bytes(secretKey.encode("ascii")),
                bytes(text.encode("ascii")),
                hashlib.sha256,
            ).hexdigest()
            headers = {
                "API-KEY": apiKey,
                "API-TIMESTAMP": timestamp,
                "API-SIGN": sign,
            }
            time.sleep(0.2)
            res = requests.post(
                endPoint + path, headers=headers, data=json.dumps(reqBody)
            )
            tmp = res.json()

            if tmp["status"] == 1 and tmp["messages"][0]["message_code"] == "ERR-5009":
                timestamp = "{0}000".format(
                    int(time.mktime(datetime.now().timetuple()))
                )
                text = timestamp + method + path + json.dumps(reqBody)
                sign = hmac.new(
                    bytes(secretKey.encode("ascii")),
                    bytes(text.encode("ascii")),
                    hashlib.sha256,
                ).hexdigest()
                headers = {
                    "API-KEY": apiKey,
                    "API-TIMESTAMP": timestamp,
                    "API-SIGN": sign,
                }
                time.sleep(0.2)
                res = requests.post(
                    endPoint + path, headers=headers, data=json.dumps(reqBody)
                )
                tmp = res.json()
        if res.json()["data"]["success"] != []:
            logger.info(json.dumps(res.json(), indent=2))
        return res

    def get_active_orders(self, symbol):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/activeOrders"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        parameters = {"symbol": symbol}

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers, params=parameters)
        return res.json()

    def get_position(self, symbol):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/positionSummary"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        parameters = {"symbol": symbol}

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers, params=parameters)

        cnt = 0
        while "data" not in res.json():
            time.sleep(0.2)
            res = requests.get(endPoint + path, headers=headers, params=parameters)
            cnt += 1
            if cnt == 10:
                raise TimeoutError

        position = {"buy": 0.0, "sell": 0.0}

        if len(res.json()["data"]["list"]) > 0:
            positions = res.json()["data"]["list"]
            for p in positions:
                if p["side"] == "SELL":
                    position["sell"] = float(p["sumPositionQuantity"])
                elif p["side"] == "BUY":
                    position["buy"] = float(p["sumPositionQuantity"])

        return position

    def get_position_rate(self, symbol):

        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/positionSummary"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        parameters = {"symbol": symbol}

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers, params=parameters)

        cnt = 0
        while "data" not in res.json():
            time.sleep(0.2)
            res = requests.get(endPoint + path, headers=headers, params=parameters)
            cnt += 1
            if cnt == 10:
                raise TimeoutError

        position = {"buy": 0.0, "sell": 0.0}

        if len(res.json()["data"]["list"]) > 0:
            positions = res.json()["data"]["list"]
            for p in positions:
                if p["side"] == "SELL":
                    position["sell"] = float(p["averagePositionRate"])
                elif p["side"] == "BUY":
                    position["buy"] = float(p["averagePositionRate"])

        return position

    def get_position_assets(self):
        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/account/assets"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers)
        data = res.json()["data"]
        for d in data:
            if d["symbol"] == "JPY":
                return int(d["amount"])

    def get_latest_execution(self):
        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/latestExecutions"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        parameters = {"symbol": "BTC_JPY", "page": 1, "count": 100}

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers, params=parameters)
        df = pd.json_normalize(res.json()["data"]["list"])
        df["size1"] = df["size"].astype(float)
        df["price"] = df["price"].astype(float)
        df["lossGain"] = df["lossGain"].astype(float)

        df = df[
            ["price", "lossGain", "settleType", "side", "size1", "timestamp", "orderId"]
        ]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.tz_convert("Asia/Tokyo")

        settleType_array, side_array, size_array = (
            df.settleType.values,
            df.side.values,
            df.size1.values.tolist(),
        )
        initial_settleType = settleType_array[0]
        initial_side = side_array[0]
        total_size = size_array[0]
        orders = set([df.orderId.values.tolist()[0]])
        for settletype, side, size, order in zip(
            settleType_array[1:],
            side_array[1:],
            size_array[1:],
            df.orderId.values.tolist()[1:],
        ):
            if settletype == initial_settleType and side == initial_side:
                total_size += size
                orders.add(order)
            else:
                break
        return round(total_size, 2), initial_settleType, initial_side, list(orders)

    def get_latest_execution_df(self):
        timestamp = "{0}000".format(int(time.mktime(datetime.now().timetuple())))
        method = "GET"
        endPoint = "https://api.coin.z.com/private"
        path = "/v1/latestExecutions"

        text = timestamp + method + path
        sign = hmac.new(
            bytes(secretKey.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        parameters = {"symbol": "BTC_JPY", "page": 1, "count": 80}

        headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
        time.sleep(0.2)
        res = requests.get(endPoint + path, headers=headers, params=parameters)
        while "data" not in res.json():
            time.sleep(0.3)
            res = requests.get(endPoint + path, headers=headers, params=parameters)
            if "data" in res.json():
                break
            time.sleep(0.3)
            timestamp = "{0}000".format(
                td + int(time.mktime(datetime.now().timetuple()))
            )
            headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
            res = requests.get(endPoint + path, headers=headers, params=parameters)
            time.sleep(0.3)
            if "data" in res.json():
                break
            timestamp = "{0}000".format(
                int(time.mktime(datetime.now().timetuple()))-td
            )
            headers = {"API-KEY": apiKey, "API-TIMESTAMP": timestamp, "API-SIGN": sign}
            res = requests.get(endPoint + path, headers=headers, params=parameters)
            if "data" in res.json():
                break
            cnt += 1
            if cnt == 20:
                raise TimeoutError
        if "list" in res.json()["data"]:
            df = pd.json_normalize(res.json()["data"]["list"])
            df["size_"] = df["size"].astype(float)
            df["price"] = df["price"].astype(float)
            df["lossGain"] = df["lossGain"].astype(float)

            df = df[
                [
                    "price",
                    "lossGain",
                    "settleType",
                    "side",
                    "size_",
                    "timestamp",
                    "orderId",
                    "positionId",
                ]
            ]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df = df.tz_convert("Asia/Tokyo")

        else:
            df = pd.DataFrame(
                columns=[
                    "price",
                    "lossGain",
                    "settleType",
                    "side",
                    "size_",
                    "timestamp",
                    "orderId",
                    "positionId",
                ]
            )
        return df
