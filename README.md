# getpara

  main.py 
    rawdataの解析用
    option 
    -a : 全てのパルスを解析
    -t : テストモード　ランダムで任意の長さのrawdataを抽出して解析
    -p : PoST解析用　チャンネル毎のsettingを変えて解析
    -l : lowpass filter
    -f : fitting 

  解析パラメータはsetting.jsonファイルで変更
  
  "Config": {
        "path": "E:/tsuruta/20230616_post/room1-ch2-3_180mK_570uA_100kHz_g10",　#データパス
        "channel": "0", #解析チャンネル
        "rate": 1000000,　#サンプルレート
        "samples": 100000, #サンプル数/チャンネル
        "presamples": 10000, #トリガー点
        "threshold": 0.05,　# 閾値
        "output": "lpf5000"　#outputフォルダーの名前
    },
    "main": {
        "base_x": 1000, #ベースラインを決めるスタート点　presamples - base_x　を開始地点とする
        "base_w": 500,　#ベースラインを決める幅　base_wの間のサンプルの平均値をベースラインとする
        "peak_max": 1000,　#ピークを決める範囲　presamplesからpeak_maxの幅の範囲内の最大値をピークとする
        "peak_x": 3, #平均ピークをとる幅のスタート地点　peak_maxから-peak_x地点を範囲のスタート地点とする
        "peak_w": 10,　#平均ピークを決める幅　peak_xをスタートとしてpeak_wの幅の範囲内の平均値を平均ピークとする
        "fit_func": "monoExp",　#
        "fit_x": 5000,
        "fit_w": 50000,
        "fit_p0": [0.1,1e-5],
        "mv_w": 1,
        "cutoff": 10000.0
    },

    
   
