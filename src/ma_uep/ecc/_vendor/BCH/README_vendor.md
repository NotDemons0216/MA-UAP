# Vendored BCH mother code

来源：本仓库开发期的 `BCH/` 目录（16 个 .py，约 7,074 行）。
MA-UEP 不修改母码实现，仅按原样打包，以便 `git clone` 后无需外部目录即可运行。

- 位序约定：低次优先，`codeword[0]` 为最低次系数；系统编码后校验位在低次、数据位在高次。
- 依赖：仅 Python 标准库。
- 引用方式：`ma_uep.ecc.vendor` 会把本目录加入 `sys.path`，随后 `from bch import BCHCode`。

如需使用别的 BCH 实现，设置环境变量 `MA_UEP_BCH_PATH` 指向包含 `bch.py` 的目录即可覆盖。
