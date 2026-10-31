# 第三者のライブラリ・資料

依存ライブラリ本体や公式サンプルデータはこのGitリポジトリに同梱しません。
バージョン・配布元URL・配布物ハッシュは`uv.lock`に記録し、各配布元のライセンスが適用されます。
ライブラリのライセンスが、その取得先の金融データを再配布する許諾になるわけではありません。

- [Microsoft Qlib](https://github.com/microsoft/qlib)：pyqlib 0.9.7。公式converterを使う外部検証は
  実行時ダウンロードで、取得ファイルと生成データはGit対象外です。
- [LightGBM](https://github.com/microsoft/LightGBM)：4.6.0。
- [yfinance](https://github.com/ranaroussi/yfinance)：0.2.66、非公式Yahooクライアント。
- 数値計算・統計・可視化・Jupyter等の全依存は`pyproject.toml`と`uv.lock`を参照。
- 複数期間寄与の参考：[Carino実装](https://github.com/R-Finance/PortfolioAttribution/blob/master/R/Carino.R)、
  [CFAの複数期間アトリビューション論文](https://rpc.cfainstitute.org/research/financial-analysts-journal/2004/multiperiod-arithmetic-attribution)。
  本実装は日次対数差の恒等式と比較経路のCAGR差を独自に実装し、参考実装本体は転載していません。
- 研究選択の限界：[Novy-Marx, Backtesting Strategies Based on Multiple Signals](https://www.nber.org/papers/w21329)。

この一覧は利用箇所・出典の案内で、全依存のライセンスに対する法律上の適合性証明ではありません。
