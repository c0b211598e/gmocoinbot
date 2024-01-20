# 使用方法

## config の記入

`apikey`、`secretKey`、証拠金のうち使用する割合（`available_margin`）を`src/config.py`に記入。

### 例

```python
class GMOBotConfig:
    apiKey = "XXXXXXX"  # GMOコインのAPIキー
    secretKey = "YYYYYYYYY"  # GMOコインのシークレットキー
    feature_pkl_path = "/work/features/features_default.pkl"  # featureのリストをpickle化したもの
    available_margin = 0.5  # 証拠金の何割をボットに使うか。0-0.85 (例 0.5 は証拠金の50%をボットに使用)
    symbol = "BTC_JPY"
    pips = 1
```

## すべてのボットの起動＋取引開始

docker-compose が使えない場合は前もってインストールをお願い致します。

```bash
docker-compose up
```

permission 関係でエラーが出る場合は

```
sudo docker-compose up
```

以上のどちらかのコマンドで Ubuntu 環境または Windows WSL2 でうまくいくはずです。

### Mac M1 で docker のコンテナ環境構築に失敗する場合

talib 周りのインストールが Mac M1でうまくいかない場合があります。

```bash
bash start_allbots_on_mac.sh
```

## 取引中断及びボットを終了する場合

`ctrl + C`で docker-compose を終了させます。

## モデルファイルの命名規則と配置場所について

モデルは`model_buy`と`model_sell`のそれぞれにに joblib で圧縮したファイルを置きます。その際の命名方法は、それぞれ、`buy_<ATR指値の係数>.xz`, `sell_<ATR指値の係数>.xz`とします。例えばチュートリアルのデフォルト通り ATR 指値の係数の 0.5 を使う場合は、`buy_0p5.xz`, `sell_0p5.xz`として訓練済みモデルをそれぞれのディレクトリ内に置きます。こうすることでボットは自動的に ATR 指値の値をファイル名から読み取り、モデルごとに異なる指値で執行できます。

## 特徴量の pkl ファイル関して

特徴量が list の型（`List[str]`）で保存されています。
