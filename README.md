# XRS 光谱处理

把原来 `XRS_spectrum.ipynb` 那套 Jupyter 流程改成**脚本 + YAML 参数**：
不再依赖 JupyterLab / ipykernel / ipycanvas / ipywidgets，改成一条命令行、
分阶段运行，缺参数时弹出 matplotlib 窗口让你调、调完自动写回 YAML。

## 快速开始

```bash
# Linux 服务器（conda，不用 pixi）
conda env create -f environment.yml
conda activate xrs
python run_xrs.py config.yaml

# 本地 Windows 开发（pixi）
pixi run python run_xrs.py config.yaml
```

`config.yaml` 初始时所有参数都是空的（`null` / `[]`），程序会在跑到对应
阶段时逐个问你，并把答案写回同一个文件（**注释和键顺序都会保留**）。

## 命令

```bash
python run_xrs.py config.yaml                   # 从第一个未完成的阶段跑起
python run_xrs.py run config.yaml --stage sum   # 只跑一个阶段
python run_xrs.py run config.yaml --from xrs    # 从某个阶段往后跑
python run_xrs.py run config.yaml --no-ui       # 不交互，只用 YAML 里已填好的值
python run_xrs.py run config.yaml --force       # 忽略参数指纹，强制重算
python run_xrs.py run config.yaml --overwrite   # 允许覆盖已存在的输出文件
python run_xrs.py check config.yaml             # 体检：还缺什么、哪些阶段要重跑
python run_xrs.py pick-roi    config.yaml --detector lambda   # 鼠标点选中心点
python run_xrs.py propose-roi config.yaml --detector lambda   # 峰值检测给候选
```

`--force` 和 `--overwrite` 是两件事：前者只管"重算"，后者只管"能否覆盖旧输出"。

## 五个阶段

| 阶段 | 做什么 | 会弹窗调的参数 |
|---|---|---|
| `elastic` | 读弹性峰扫描 → 建 ROI → 拟合弹性峰 | `filter_value`、`roi_size`、拟合窗口 |
| `xrs` | 读 XRS 扫描 → 去 I0 glitch → 各 ROI 谱 | 点曲线剔除坏扫描 |
| `q` | 动量转移 Q | 模组角度（终端问答） |
| `sum` | 能量内插与叠加 | 模组、q 范围、内插步长 |
| `save` | 写 `_data.txt` / `_rois.txt` / `_info.txt` | 输出目录名 |

每个阶段的中间结果存在 `<data.root>/processed/.xrs_state/`，QC 图存在
`.xrs_state/figures/`。**改了 `sum` 的参数只会重跑 `sum` 和 `save`，不会
重新读 HDF5。** 参数一变，下游阶段的指纹就失效，自动重算。

## 自动选取 ROI 是怎么处理的

原 notebook 里那个 ipycanvas 点选控件其实**从未被执行**：`auto` 模式
总是先从 `ROI/auto_ROI_*.txt` 读中心点，再交给 `select_roi_centers`
之外的代码做分割。所以"自动选 ROI"本来就是一条无头、可复现的流水线：

```
中心点文件 → 每个中心做局部 Otsu 阈值 → 取连通域 → 重叠像素按最近中心归属
```

现在把它明确成两件事：

1. **分析主流程完全无头**。`config.yaml` 只*引用*中心点文件：

   ```yaml
   elastic:
     roi_mode: auto
     auto_centers:
       lambda: roi/auto_ROI_lambda.txt
       minipix: roi/auto_ROI_minipix.txt
   ```

   中心点文件是"几何校准"的产物，和分析参数解耦，可以跨 beamtime 复用。
   格式是三列 `roi_label x y`（空格/制表符/逗号分隔）。

2. **几何变了才重建，用独立子命令**，不污染分析流程：

   - `pick-roi`：弹出图像，按规范标签顺序鼠标点选，右键撤销，回车完成。
     覆盖前自动备份成 `.bak`，并输出一张中心点叠加图供核对。
     （这是替代 ipycanvas 控件的方案，需要图形界面。）
   - `propose-roi`：用 `skimage.feature.peak_local_max` 找峰，优先与旧中心点
     文件做一对一最优匹配（探测器轻微漂移时只需人工修正个别点）。
     **默认只写 `*.proposed.txt`**，核对无误后再加 `--write` 覆盖正式文件。

运行时还会做几项校验，避免静默出错：

- 中心点文件里出现规范标签之外的标签 → 直接报错（多半是拼写错误）。
- 有标签没被分割出来 → 日志警告，并画在 QC 图上。
- 某个中心分割失败 → 连原因一起记进 `meta.json`、画在 QC 图上；
  `elastic.auto_params.on_roi_failure: error` 可以让它直接中止。
- `--no-ui` 下也会输出全部 QC 图，无头运行同样能核对。

## 相比 notebook 修正的问题

| # | 原 notebook 的问题 | 现在 |
|---|---|---|
| 1 | `scanIDs2` 从未定义，却被写进 info 文件 → `NameError` | 用 `xrs.scan_ids` |
| 2 | 保存时用的是另一段代码算出的 `interp_added` 配重新生成的 `E_interp` | 插值只算一次，存什么就算什么 |
| 3 | 绘图分支里访问只有两个元素的 `axs[2]` → `IndexError` | 去掉该死分支 |
| 4 | `os.mkdir` / `to_csv(mode='x')` 在目录或文件已存在时直接崩 | 清晰的报错 + `--overwrite` |
| 5 | 能量 PV 的 `try/except` 在赋值前就访问变量，导致**永远**走 fallback | 对已读入的电机表做真探测，并记录实际用了哪个 PV |
| 6 | 模组筛选规则在 cell 34 和 cell 36 里不一致（后者还永久排除了 `HR`） | 统一用 `sum.modules`，被忽略的模组会给出警告 |
| 7 | 全篇裸 `except:` | 换成明确的异常类型 |
| 8 | `num_pixel_list` / `stacked_roi_list` 赋值后从未使用 | 删除 |

**有一处行为刻意保持不变**：`q.elastic_energy_source` 默认是 `as_before`，
即复刻原 notebook 的语义——好拟合用标称的 9.685 keV，只有 bad fit 才用
拟合中心。这看起来像 bug（正常应该一律用拟合中心），但改动会直接影响
物理结果，所以保留原行为并留了 `nominal` / `fitted` 两个可选项。
**请确认哪个才是你的本意。**

## 环境注意事项

`pixi.toml` 和 `environment.yml` 的包列表必须一致，`check` 子命令会检查。

> **`libblas=*=*openblas` 这一行不要删。** conda-forge 的 scipy 是针对
> OpenBLAS 构建的。如果环境里混进 MKL 变体的 `libblas`，
> `scipy.optimize.curve_fit`（MINPACK）会以 Windows 延迟加载失败
> `0xC06D007F` **直接崩掉整个进程**，Python 层连异常都捕不到。

## 测试

```bash
pytest tests -q
```

测试用一份合成的 NeXus 数据（`tests/conftest.py` 里现造），覆盖 regular/auto
两种 ROI 模式、断点续跑、参数校验、以及交互层（把事件循环替换成模拟点击，
所以无显示器也能测）。

注意：测试刻意不用 pytest 的 `tmp_path`——它内部用 `tempfile.mkdtemp`，
会创建 `0700` 权限的目录，在受限沙箱下随后不可写。
