# 基于demucs和so-vits-svc的语音分离与生成

## 功能
- 语音分离
- 语音生成

## 安装
1. 安装一个包管理器
   - Windows: 
       ```powershell
       Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
       ```
   - Mac:
       ```bash
       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
       ```
     
2. 从[Anaconda官网](https://www.anaconda.com/download/)下载anaconda或miniconda

4. 安装并激活环境
    > 如果是macos, 先运行scripts中的`fix_mac_certificate_issue.sh`
    ```bash
    conda env create -f environment.yml
    conda activate song-generating-pack
    ```
   
4. 手动安装pytorch依赖
    windows/linux:
    ```bash
    conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia
    ```
    macos:
    ```bash
    conda install pytorch-nightly::pytorch torchaudio -c pytorch-nightly
    ```
   
5. 安装ffmpeg
    > 在运行这一步之前, 请务必安装包管理器
    > Windows请使用chocolatey, 而不是scoop, scoop没有这个包
    - Windows: `choco install ffmpeg`
    - Linux: `sudo apt install ffmpeg`
    - Mac: `brew install ffmpeg`

6. 安装CUDA release 12.1和cuDNN v8.9.2(如果你想要使用NvidiaGPU训练)
    > 不支持AMD GPU
    > 安装cudnn时会被要求注册nvidia developer account
    > CUDA和cuDNN的配置详见[这篇文章(win)](https://zhuanlan.zhihu.com/p/99880204)和[这篇文章(linux)](https://blog.csdn.net/qq_40263477/article/details/105132822)
    - Windows: [CUDA](https://developer.nvidia.com/cuda-downloads) [cuDNN](https://developer.nvidia.com/cudnn)
    - Linux: [CUDA](https://developer.nvidia.com/cuda-downloads) [cuDNN](https://developer.nvidia.com/cudnn)

## roadmap
- [ ] 文字转语音并输出
- [ ] 集成模型训练
- [ ] Qt GUI
- [ ] WebUI