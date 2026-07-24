# ChinaVoices Challenge 推理说明

本项目包含 ChinaVoices Challenge 的两个任务：

* **Task 1：语种识别（Language Identification, LID）**
* **Task 2：语音识别（Automatic Speech Recognition, ASR）**

两个任务均提供一键运行脚本。运行脚本后，项目代码和模型权重会自动下载到指定的项目运行目录中。

项目地址：

```text
https://github.com/ichi131/Chinavoice_code
```

---

## 数据准备

数据路径需要指向评测数据集的 `evaluation_set` 目录。

`evaluation_set` 目录下必须包含：

* `wav.scp`：音频索引文件
* `wav/`：存放评测音频文件的目录

推荐的数据目录结构如下：

```text
evaluation_set/
├── wav.scp
└── wav/
    ├── eval_000001.wav
    ├── eval_000002.wav
    └── ...
```

`wav.scp` 中每一行包含两个字段：

```text
<音频ID> <音频文件路径>
```

示例：

```text
eval_000001 evaluation_set/wav/eval_000001.wav
eval_000002 evaluation_set/wav/eval_000002.wav
```

其中：

* 第一个字段为音频 ID，例如 `eval_000001`。
* 第二个字段为音频文件路径，例如 `evaluation_set/wav/eval_000001.wav`。
* 两个字段之间使用空格分隔。
* 每个音频文件对应一行记录。

在下面的运行命令中：

* Task 1 的 `DATA_ROOT` 必须指向 `evaluation_set` 目录。
* Task 2 的 `EVALUATION_SET_DIR` 必须指向 `evaluation_set` 目录。

---

## Task 1：语种识别

### 1. 下载运行脚本

从项目仓库下载 `run_lid.sh`：

```bash
wget https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_lid.sh
```

也可以使用：

```bash
curl -O https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_lid.sh
```

### 2. 一键运行

```bash
WORKDIR=/n/work6/yiwang/docker_test \
DATA_ROOT=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
bash run_lid.sh
```

### 3. 参数说明

* `WORKDIR`：项目运行目录。项目代码和模型权重会自动下载到该目录下。
* `DATA_ROOT`：评测数据目录，必须指向 `evaluation_set` 目录。该目录下需要包含 `wav.scp` 和 `wav/` 音频目录。

运行后，项目默认位于：

```text
${WORKDIR}/Chinavoice_code
```

### 4. 输出结果

语种识别结果保存在：

```text
${WORKDIR}/Chinavoice_code/infer_data/lid.jsonl
```

以上述运行命令为例，实际输出路径为：

```text
/n/work6/yiwang/docker_test/Chinavoice_code/infer_data/lid.jsonl
```

---

## Task 2：语音识别

### 1. 下载运行脚本

从项目仓库下载 `run_asr.sh`：

```bash
wget https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_asr.sh
```

也可以使用：

```bash
curl -O https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_asr.sh
```

### 2. 一键运行

```bash
CLONE_DIR=/n/work6/yiwang/docker_test \
EVALUATION_SET_DIR=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
bash run_asr.sh
```

### 3. 参数说明

* `CLONE_DIR`：项目运行目录。项目代码和模型权重会自动下载到该目录下。
* `EVALUATION_SET_DIR`：评测数据目录，必须指向 `evaluation_set` 目录。该目录下需要包含 `wav.scp` 和 `wav/` 音频目录。

运行后，项目默认位于：

```text
${CLONE_DIR}/Chinavoice_code
```

### 4. 输出结果

语音识别结果保存在：

```text
${CLONE_DIR}/Chinavoice_code/FireRedASR2S-fintuning/exp/asr_chinavoices_eval/asr.jsonl
```

以上述运行命令为例，实际输出路径为：

```text
/n/work6/yiwang/docker_test/Chinavoice_code/FireRedASR2S-fintuning/exp/asr_chinavoices_eval/asr.jsonl
```

---

## 完整运行示例

### Task 1

```bash
wget https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_lid.sh

WORKDIR=/n/work6/yiwang/docker_test \
DATA_ROOT=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
bash run_lid.sh
```

输出文件：

```text
/n/work6/yiwang/docker_test/Chinavoice_code/infer_data/lid.jsonl
```

### Task 2

```bash
wget https://raw.githubusercontent.com/ichi131/Chinavoice_code/main/run_asr.sh

CLONE_DIR=/n/work6/yiwang/docker_test \
EVALUATION_SET_DIR=/n/work6/yiwang/chinavoices_challenge/evaluation_set \
bash run_asr.sh
```

输出文件：

```text
/n/work6/yiwang/docker_test/Chinavoice_code/FireRedASR2S-fintuning/exp/asr_chinavoices_eval/asr.jsonl
```

---

## 注意事项

1. `DATA_ROOT` 和 `EVALUATION_SET_DIR` 都需要指向 `evaluation_set` 目录，而不是其上一级目录。
2. `evaluation_set` 目录下必须包含 `wav.scp` 文件和 `wav/` 音频目录。
3. 请确保 `wav.scp` 中记录的音频路径与实际文件位置一致。
4. 请确保指定的项目运行目录具有足够的磁盘空间和读写权限。
5. 首次运行时，脚本需要下载项目代码、Docker 镜像和模型权重，因此需要可用的网络连接。
6. 如果项目运行目录中已经存在旧版本的 `Chinavoice_code`，脚本可能会更新或复用已有目录，请提前备份重要文件。
