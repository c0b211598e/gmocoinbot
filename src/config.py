class GMOBotConfig:
    apiKey = "XXXXXXX"  # GMOコインのAPIキー
    secretKey = "YYYYYYYYY"  # GMOコインのシークレットキー
    feature_pkl_path = "/work/features/features_default.pkl"  # featureのリストをpickle化したもの
    available_margin = 0.5  # 証拠金の何割をボットに使うか。0-0.85 (例 0.5 は証拠金の50%をボットに使用)
    symbol = "BTC_JPY"
    pips = 1
