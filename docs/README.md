# 文档索引

本目录记录花椒含油率检测项目的工程交付、实验记录和运行手册。当前应用集成优先阅读以下文档：

## 应用集成与交付

1. `../README.md`
   - 顶部 Quickstart 给出当前 DINOv3 ROI + gray_otsu refined ROI + 含油率部署候选模型的 CLI/API 用法。
2. `../ARCHITECTURE.md`
   - 端到端数据流、artifact 文件契约、模块职责和安全约束。
3. `specs/huajiao-oil-model-serving.md`
   - 核心 predictor 输入/输出 JSON schema、错误码、验收标准、非目标。
4. `specs/huajiao-web-app.md`
   - Web/FastAPI 上传接口、响应 JSON、可视化 URL、输出目录和验收标准。
5. `runbooks/huajiao-inference.md`
   - 单样本 CLI/Python 推理、真实样本 smoke、debug 输出和常见问题排查。
6. `runbooks/huajiao-web-app.md`
   - Web 服务启动、curl/TestClient smoke、部署排查和质量门。
7. `runbooks/model-release.md`
   - 重新提取 DINOv3 refined ROI 特征、打包 artifact、校验、发布和回滚。

## 当前发布/候选 artifact

| Artifact | 类型 | 状态 |
| --- | --- | --- |
| `artifacts/huajiao-oil-content-dinov3-gray-otsu-refined-roi-v0.3.1` | 含油率部署候选模型 | 当前推荐；RGB + ENVI cube；DINOv3 coarse polygon + gray_otsu refined ROI |
| `artifacts/huajiao-oil-content-dinov3-roi-v0.3.0` | 含油率部署对照模型 | DINOv3 coarse polygon 口径；coverage 过大，已被 v0.3.1 refined 取代为首选候选 |
| `artifacts/huajiao-dinov3-roi-seg-v0.1.0` | ROI 分割模型 | 当前 DINOv3 coarse polygon 后端 |
| `artifacts/huajiao-report-20260505-snv-sg1-pls10-batch-quality-huber` | 精确复现/内部对照 | 不可作为只上传 RGB+cube 的应用部署模型 |

## 关键边界

精确复现模型和部署模型不是同一个输入约束：

- 精确复现模型为了复现报告指标，包含 `weight_g`、`batch`、`sid_num`。
- 应用展示部署只上传 RGB 图像和 ENVI cube，无法提供重量、批次、样本编号等实验上下文字段。
- 因此当前交付模型必须只使用上传文件可直接得到的 `wl_*` ROI 平均光谱和 ROI 质量特征。
- DINOv3 coarse polygon 分割 Dice 高不等于下游含油率一定好；当前 v0.3.1 先用 DINOv3 定位 coarse polygon，再在 polygon 内做 gray_otsu 二次精分割，使 ROI coverage 回到 0.16-0.18 的合理区间。

## 参考资料与实验记录

- `reference_docs/功能需求文档/含油量检测需求.md`
- `reference_docs/实验记录/花椒含油率模型复现记录_20260509.md`
- `reference_docs/实验记录/花椒ROI_DINOv3分割微调实验报告_20260509.md`
- `reference_docs/实验记录/花椒含油率DINOv3二次精分割实验记录_20260509.md`

这些记录可用于追溯实验过程，但应用调用、验收和发布应以 `specs/` 与 `runbooks/` 为准。
