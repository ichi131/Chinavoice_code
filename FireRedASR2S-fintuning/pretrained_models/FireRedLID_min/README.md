离线精简版 LID 推理资产，从 `pretrained_models/FireRedLID` 抽取，只服务于
`examples_train/lid_chinavoices/{finetune,infer}_lid_chinavoices.py` 这条 LID 分类链路，
使解码/评测不再依赖体积几 GB 的原始 `pretrained_model_dir`。

包含两个文件：
- `model.pth.tar`：只保留原文件里的 `args`（模型结构配置，如 `d_model`/`n_layers_enc` 等），
  `model_state_dict` 是空字典。`LidClassifier` 用 `args` 搭建 encoder 骨架后，真正的权重
  会被 finetune 出来的 checkpoint 以 `strict=True` 整体覆盖，因此这里的权重值本来就不会被用到。
- `cmvn.ark`：原样拷贝自 `pretrained_models/FireRedLID/cmvn.ark`，特征归一化统计量，
  必须和训练时用的一致，不能重新生成。

如果 `pretrained_models/FireRedLID` 结构配置发生变化（比如换了新底座模型），用
`python3.10 examples_train/lid_chinavoices/make_lid_min_assets.py` 重新生成本目录。
