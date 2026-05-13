# 花椒 ROI DINOv3 分割模型

- 模型: dinov3_vits16 + lightweight binary segmentation head
- Backbone 权重: /home/yr/yr/code/cv/large_models/dinov3_all/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
- 训练 split: 当前含油率 train，样本数 157
- 测试/验证 split: 当前含油率 val，样本数 40
- 输入: RGB 图片，训练时 resize 到 448x448
- 输出: 二值花椒 ROI mask，默认阈值 0.5
- 最佳 epoch: 18
- 最佳 val Dice: 0.975292

注意：artifact 不包含原始训练数据和 DINOv3 backbone 权重；部署环境需要能访问 config.json 中记录的本地权重路径，或由发布流程同步该权重。
