# ChinaVoices Challenge 推理说明

本项目包含 ChinaVoices Challenge 的两个任务：

* **Task 1：语种识别（Language Identification, LID）**
* **Task 2：语音识别（Automatic Speech Recognition, ASR）**

两个任务均提供一键运行脚本。运行脚本后，项目代码和模型权重会自动下载到指定的工作目录中。

项目地址：

```text
https://github.com/ichi131/Chinavoice_code
```

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
* `DATA_ROOT`：评测数据集所在目录。

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
* `EVALUATION_SET_DIR`：评测数据集所在目录。

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

1. 请确保指定的工作目录具有足够的磁盘空间和读写权限。
2. 请确保评测数据目录填写正确。
3. 首次运行时，脚本需要下载项目代码、Docker 镜像及模型权重，因此需要可用的网络连接。
4. 如果目标工作目录中已经存在旧版本的 `Chinavoice_code`，脚本可能会更新或复用已有目录，请根据实际情况提前备份重要文件。
