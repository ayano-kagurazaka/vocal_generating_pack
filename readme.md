# vocal generating pack

> 一个集成了语音分离, 语音生成和语音合并的工具包

## 功能
- 语音分离
- 语音生成
- 合并音频

## 安装

Nvidia显卡和CPU用户参见[环境配置](doc/environment.md)

AMD显卡用户请遗憾离场, 目前不支持AMD显卡

## demo

在配置好环境以后使用Jupyter notebook运行[src/demo.ipynb](src/demo.ipynb)即可, 目前里面包含了一个简单的inference流程和demo歌曲

## roadmap
- [ ] 文字转语音并输出
- [ ] 模型训练
- [ ] 分离单个音频中不同人的声音
- [ ] Beeware UI
- [ ] CLI
- [ ] Jupyter Notebook
  - [x] Inference
  - [ ] Training
- [ ] 资源管理器
  - [x] 模型管理器
  - [ ] 数据集管理器
- [ ] 插件系统
- [ ] AMD GPU支持

## 使用的资源一览

- [demucs(MIT)](https://github.com/facebookresearch/demucs)
- [so-vits-svc-fork(MIT/Apache 2.0)](https://github.com/voicepaw/so-vits-svc-fork)
- [原神so-vits-svc模型(MIT)](https://huggingface.co/kaze-mio/so-vits-genshin)
- [Audio Slicer(MIT)](https://github.com/openvpi/audio-slicer)
- [pyannote-audio(MIT)](https://github.com/pyannote/pyannote-audio)
- [空庭雨(已获得原作者授权)](https://music.163.com/song?id=2006730110)
- [星空逃避行](https://tandess.itch.io/escape-demo)
- [DAI☆TAN センセーション](https://www.tandess.com/en/music/free-material/material.html)
- [永遠と唄](https://www.tandess.com/en/music/free-material/material.html)
- [ディザイア](https://www.tandess.com/en/music/free-material/material.html)


