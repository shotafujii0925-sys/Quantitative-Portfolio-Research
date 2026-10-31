# Qlibの版・データ・検証

2026-10-31に公式ドキュメント、公式リリース、PyPIとローカルコードを確認した。

## 導入する版

- [GitHub公式リリース](https://github.com/microsoft/qlib/releases)はv0.9.8が最新。
  これはセキュリティ強化を含むソースリリースで、PyPI公開状況とは異なる。
- [PyPI](https://pypi.org/project/pyqlib/0.9.7/)には0.9.7のPython 3.11用Windows/Linux wheelがある。
  今回はビルドツールをシステムに導入せず利用できる0.9.7を固定した。
- [v0.9.7の依存定義](https://github.com/microsoft/qlib/blob/v0.9.7/pyproject.toml)と
  実テストに基づき、numpy 1.26.4、pandas 2.2.3、LightGBM 4.6.0、
  MLflow 3.1.0、PyArrow 20.0.0を採用。PyTorchはQlibの木モデルには不要。
- 試行したMLflow 3.17.2はローカルfile store利用に追加設定が必要で既定では失敗した。
  Qlib側にもURIをdecodeせずlockパスに使う処理があるため、日本語パスではURIを
  明示的にdecodeしている。PyArrow 21とMLflow 3.1.0は依存条件が衝突したため20へ変更。
- 0.9.7は最新ソースの全修正を含まない。外部のpickle・モデルや未知の設定ファイルを
  ロードしない。ソース最新版への更新は別環境で同じ統合テストを通してから行う。

## ネットワーク不要で起動

```powershell
.\.venv\Scripts\python.exe scripts/run_qlib.py
```

`data/qlib_synthetic`に固定seedの6資産×420営業日データを生成する。
calendars/day.txt、instruments/all.txt、features/<銘柄>/<field>.day.binができる。
binはfloat32 little-endianで、最初の数値がcalendar開始添字、以降が値。
公式dump_bin.pyと同じ格納仕様だが、任意市場のデータを変換する汎用実装ではない。

initはREG_US、kernels=1、Redisキャッシュなし。Redis/MongoDBサービスは不要。
DatasetHとDataHandlerLPを使い、過去式だけの4特徴量からLGBModelを学習する。
QlibのTopkDropoutStrategy、backtest_daily、risk_analysisを実際に実行する。
出力は`分析結果/レポート/qlib_synthetic_*.csv`、`qlib_summary.json`。

QlibにはGymのメンテナンス終了通知、未導入の任意モデル（Torch/CatBoost/XGBoost）、
future calendarへのfallback、バックテスト内の空集合平均などの警告が出る場合がある。
空集合平均は取引指標集計に由来し、ポートフォリオreportとriskの有限性はテストで確認する。
警告は抑制せず、検証ログに残す。取引を行わない日の個別取引指標は未定義になり得る。

## 公式サンプルの取得と容量

[公式データ仕様](https://qlib.readthedocs.io/en/stable/component/data.html)と
[v0.9.7 GetData実装](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/tests/data.py)を参照。
現在のmainと0.9.7では取得先・CLI案内が異なる場合があり、混在させない。

```powershell
.\.venv\Scripts\python.exe scripts/check_external.py
```

このスクリプトはGitHub release APIで圧縮サイズを照会し、
`分析結果/レポート/external_checks.json`に記録する。データアーカイブはダウンロードしない。
展開サイズはAPIから分からないため未確認。十分な空き容量を用意し、ZIP中央ディレクトリの
展開サイズも確認してから別名の空フォルダへ展開する。既存データへ重ねて展開しない。

今回の容量照会は日次フル196,549,189 bytes、簡易51,810,838 bytes。
大型データとは別に、小型CSV配布173,212 bytes（展開696,003 bytes）だけを
`scripts/check_official_sample.py`で取得し、85銘柄の変換と先頭2銘柄のQlib式評価を確認した。
元CSVにfactorがないため、rawを保持してadjclose/closeからfactorを構成している。
この事後調整は約定価格やpoint-in-timeの保証を持たず、実証的な運用評価には使用していない。

公式の例は以下。実行前に同じ版のget_data.pyを取得し、出力先が新規であることを確認する。

```text
python scripts/get_data.py qlib_data --target_dir <NEW_EMPTY_DIRECTORY> --region cn
```

GetDataのdelete_old既定値は削除を伴う場合がある。既存フォルダを指定しない。
データはYahoo等由来の参考データであり、正確性・完全性・現在性・再配布権を保証しない。
公式サンプル全体を使った学習や実市場での収益性は今回の検証対象に含まない。

## CSV/Parquetを公式ツールで変換

[v0.9.7公式converter](https://github.com/microsoft/qlib/blob/v0.9.7/scripts/dump_bin.py)
を用いる。`scripts/check_external.py`はその版のスクリプトを`分析結果/公式変換ツール/`に取得し、
SHA256を記録して合成CSVを実際に変換する。80行の価格binと元CSVが一致するか検証する。
取得したスクリプトは`分析結果/`内にだけ置きGit対象から除外する。

```text
python dump_bin.py dump_all --data_path <CSV_DIRECTORY> --qlib_dir <NEW_PROVIDER> --include_fields open,close,high,low,volume,factor --max_workers 1
```

CSVは銘柄ごと（例SYN00.csv）、date,open,close,high,low,volume,factor列。
調整価格・出来高の基準を揃える。factor=調整価格/元価格。
yfinanceの配当調整をQlibの分割調整規約へ無検証でそのまま流さない。
実データの分割・配当、売買停止時NaN、休日、上場廃止は別途検証が必要。

## PyTorch CPU（任意）

`pyproject.toml`のtorch extraに2.7.1+cpu、公式CPU indexを明示してロックしている。
現在の基礎環境にはインストールしない。wheel圧縮サイズはexternal_checks.jsonに記録する。
HEAD照会ではWindows x64版216,031,210 bytes、Linux x64版175,966,816 bytesだった。
必要容量はwheel以外に依存・展開・uvキャッシュも加算される。

```powershell
powershell -File scripts/setup.ps1 -WithTorch
```

このextraは依存解決のみ検証対象で、Torchのimportや深層モデル学習は未検証。
