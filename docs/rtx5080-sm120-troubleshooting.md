# RTX 5080 (sm_120 / Blackwell) 踩坑记录

> 记录一次在 AutoDL 容器里用 RTX 5080 跑 ColonSplat（3D Gaussian Splatting）训练时，
> 遇到 `no kernel image is available for execution on the device` 的完整排查与修复过程。
>
> 日期：2026-10-14
> 结论先行：**RTX 50 系（Blackwell，sm_120）必须用 CUDA 12.8+ 的 PyTorch（即 `+cu128` wheel），并把所有 CUDA 扩展一并重编。**

---

## 1. 环境信息

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 |
| Compute Capability | **12.0（sm_120，Blackwell 架构）** |
| 容器 | AutoDL，`/root` 工作区 |
| conda 环境 | `env`，Python 3.9，`/root/miniconda3/envs/env` |
| 系统 CUDA Toolkit | `/usr/local/cuda-12.8`（nvcc 12.8.93） |
| 项目 | `/root/ColonSplat` |

> ⚠️ 注意：`env` 环境自带的 `nvcc` 是 **12.1**（且不在 PATH 里），
> 系统里真正的 CUDA 12.8 在 `/usr/local/cuda-12.8`。编译扩展时必须显式指定后者。

---

## 2. 现象

训练命令：

```bash
cd /root/ColonSplat
python train.py -s /mnt/a-1/C3VDv2/c1_sigmoid1_t4_v4 \
  --port 6019 --expname run/test/c2_ascending_t1_v4 \
  --iterations 1000 \
  --data_width 675 --data_height 540 \
  --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 --col_smooth_weight 0.0001
```

数据一切正常加载：

```
meta data loaded, total image:297
...
Number of points at initialisation :  364500
```

随后在初始化高斯（`create_from_pcd`）时崩溃：

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

崩溃点：`scene/gaussian_model.py` 第 220 行

```python
dist2 = torch.clamp_min(
    distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
    0.0000001
)
```

即调用 `simple_knn` 扩展里的 `distCUDA2` 时触发。

其实在更早的时候，PyTorch 就已经给出了明确警告（很容易被忽略）：

```
UserWarning:
NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities
sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
If you want to use the NVIDIA GeForce RTX 5080 GPU with PyTorch,
please check the instructions at https://pytorch.org/get-started/locally/
```

---

## 3. 根因

### 3.1 直接原因

原始环境是 PyTorch **2.4.1+cu121**。它的 CUDA 内核只编译到 `sm_90`：

```bash
python -c "import torch; print(torch.version.cuda, torch.cuda.get_arch_list())"
# 12.1 ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
```

而 RTX 5080 是 `sm_120`。**没有匹配的 cubin 内核，任何 CUDA 算子都会直接报错。**

### 3.2 为什么升级到"差不多的"版本也没用

CUDA 12.1 的时代根本没有 Blackwell。要支持 `sm_120`：

| CUDA 版本 | 是否支持 sm_120 |
| --- | --- |
| 12.1 / 12.4 | ❌ 不支持 |
| 12.6 | ⚠️ 视 wheel 而定，实测拿到的 `2.7.1+cu126` 内核只到 `sm_90a` |
| **12.8** | ✅ 正式支持 Blackwell |

所以目标很明确：**PyTorch `+cu128`**。

### 3.3 为什么光升 PyTorch 还不够

`simple_knn` 和 `diff_gaussian_rasterization` 是两个**本地编译的 CUDA 扩展**。
它们是用旧 torch 2.4.1 编译的产物，有两个问题：

1. 编译时只针对旧架构（无 `sm_120`）；
2. 换了 torch 之后 ABI 不匹配（会报 `libc10.so` 找不到之类的符号错误）。

→ **必须用新 torch + CUDA 12.8 重新编译这两个扩展。**

---

## 4. 排查过程中踩的坑

### 坑 1：镜像源静默回退，装了个"假 cu128"

当时用了这个命令：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128/ \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

结果装成了 `2.7.1+cu126`。原因：

- 阿里云那个 URL 是**网页索引**，页面把文件名里的 `+` 转义成了 `&#43;`，
  pip 拿到的全是乱码文件名，解析不出任何包（`pip index versions torch` 直接报
  `No matching distribution found for torch`）；
- 于是 pip 静默回退到 `--extra-index-url` 指定的清华 PyPI，装了个默认版本。

**教训**：`--extra-index-url` 有风险 —— 主源解析失败时它会悄悄顶上，而且不报错。

实测各镜像源（`pip index versions torch`）：

```
✅ https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/   → 2.7.0/2.7.1/2.8.0+cu128
✅ https://download.pytorch.org/whl/cu128             → 同上（官方）
❌ https://mirrors.aliyun.com/pytorch-wheels/cu128/   → 解析失败
❌ https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128/ → 404
```

### 坑 2：无卡开机时 CUDA 不可用

用**无卡开机**（没有物理 GPU）调试时：

```bash
python -c "import torch; print(torch.cuda.is_available())"   # False
# RuntimeError: No CUDA GPUs are available
```

这是**正常现象**，不是环境坏了。但要注意：

- 无卡时可以检查和编译，但**无法验证**内核能否真正运行；
- 编译扩展时必须显式设置 `TORCH_CUDA_ARCH_LIST`，否则 torch 会尝试探测 GPU 然后失败。

---

## 5. 完整修复步骤

### 步骤 1：安装 cu128 版 PyTorch

```bash
conda activate env

# 先卸载装错的版本
python -m pip uninstall -y torch torchvision torchaudio

# 用上交镜像装 cu128（显式带 +cu128，避免被降级）
python -m pip install "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128" \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/ \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤 2：重新编译两个 CUDA 扩展

```bash
conda activate env
cd /root/ColonSplat

# 编译依赖
python -m pip install -U pip ninja setuptools wheel

# 清掉旧产物 + 卸载旧扩展
rm -rf submodules/simple-knn/build submodules/simple-knn/*.egg-info
rm -rf submodules/diff-gaussian-rasterization-radegs/build \
       submodules/diff-gaussian-rasterization-radegs/*.egg-info
python -m pip uninstall -y simple_knn diff_gaussian_rasterization

# 指向 CUDA 12.8（关键！env 自带的是 12.1）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
# 无卡开机时必须显式指定；有卡时写上也无害
export TORCH_CUDA_ARCH_LIST="12.0+PTX"

nvcc --version | tail -2          # 必须显示 release 12.8
# 编译（各需几分钟，无输出不代表卡住）
python -m pip install --no-build-isolation ./submodules/simple-knn
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization-radegs
```

### 步骤 3：验证（换有卡开机后）

```bash
conda activate env
cd /root/ColonSplat

# torch 侧
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torch; print('sm_120' in torch.cuda.get_arch_list())"

# 内核侧（真实调用）
python -c "import torch; from simple_knn._C import distCUDA2; \
           x=torch.rand(1000,3).cuda(); print('distCUDA2 ok', distCUDA2(x).shape)"
python -c "import diff_gaussian_rasterization as d; print('rasterizer ok', d.__file__)"
```

### 步骤 4：重跑训练

```bash
cd /root/ColonSplat
python train.py -s /mnt/a-1/C3VDv2/c1_sigmoid1_t4_v4 \
  --port 6019 --expname run/test/c2_ascending_t1_v4 \
  --iterations 1000 \
  --data_width 675 --data_height 540 \
  --depth_weight 0.25 --knn_weight 0.01 --dcol_weight 0.005 --col_smooth_weight 0.0001
```

---

## 6. 修复后的实际结果

```
torch            2.7.1+cu128
torch.version.cuda  12.8
libtorch_cuda.so 内核:  sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_120

simple_knn._C.so              内核:  sm_120
diff_gaussian_rasterization._C.so 内核:  sm_120
```

两个扩展都成功编成了 `sm_120`。

---

## 7. 方法论：如何自己判断"版本对不对"

这是本次最有价值的经验 —— **不要只看版本号，要看内核架构**。三条命令，无需 GPU：

### 7.1 看 torch 声称支持的架构

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

### 7.2 直接解剖 wheel 里编译好的内核（最可靠）

```bash
SO=$(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libtorch_cuda.so'))")
/usr/local/cuda-12.8/bin/cuobjdump --list-elf "$SO" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u
```

只要输出里**没有** `sm_120`，就说明这个 wheel 在 5080 上跑不了，
无论它自称 `+cu126`、`+cu128` 还是别的。

### 7.3 检查本地扩展编了哪些架构

```bash
find /root/miniconda3/envs/env/lib/python3.9/site-packages -name "*_C*.so" \
  -path "*simple_knn*" -o -name "*_C*.so" -path "*diff_gaussian*" \
| while read so; do echo "-- $so"; \
  /usr/local/cuda-12.8/bin/cuobjdump --list-elf "$so" | grep -oE 'sm_[0-9]+[a-z]?' | sort -u; done
```

---

## 8. 备忘清单（换机器 / 重建环境时照抄）

- [ ] GPU 是 50 系（Blackwell / sm_120）？→ 必须用 `+cu128`
- [ ] 装 PyTorch 时 `--index-url` 指向 **上交镜像或官方**，别用阿里云网页索引
- [ ] 装完立刻用 `cuobjdump` 确认内核含 `sm_120`，**不要只看版本号**
- [ ] `CUDA_HOME=/usr/local/cuda-12.8`，确认 `nvcc --version` 是 12.8
- [ ] `TORCH_CUDA_ARCH_LIST="12.0+PTX"`（无卡开机时必须设）
- [ ] 换 torch 后，`simple-knn` 和 `diff-gaussian-rasterization-radegs` **必须重编**
- [ ] 重编前删除 `build/` 与 `*.egg-info`，避免复用旧对象文件
- [ ] 编译用 `--no-build-isolation`，确保用的是新装的 torch

---

## 附：同期遇到的其他问题（非本次核心）

排查 sm_120 的过程中，训练脚本还暴露了几个数据侧问题，一并记录：

1. **目录命名不匹配** —— 读取器（`scene/endo_loader.py`）硬编码查找 `color/` 和 `depth/`，
   而 C3VDv2 v4 数据是 `rgb/` 和 `depth_dam/`，需重命名或建立软链接。
   断言 `len(image_paths) == poses.shape[0]` 就是在这里失败。

2. **需要 `pose.txt`** —— C3VD 读取器以 `pose.txt` 的存在为分派条件，
   每行 16 个逗号分隔的数值（4×4 外参矩阵）。

3. **分辨率硬编码** —— 内参参考分辨率写死为 675×540。
   已在 `scene/dataset_readers.py`、`scene/endo_loader.py`、`scene/__init__.py`、
   `arguments/__init__.py` 中新增 `--data_width` / `--data_height`，
   读取器会按 `scale = image_width / 675` 等比缩放 `fx/fy/cx/cy`，
   并对 color / depth 做 resize（depth 用 `INTER_LINEAR`，并 squeeze 掉多余维度）。

4. **深度图需自行估计** —— v4 序列没有 GT depth，需先用 `pre_dam_dep.py`
   （Depth-Anything ONNX）生成深度图。
   注意 `--rgb_paths` 要传**相对路径** `rgb`，传绝对路径会导致输出目录算错。
   ONNX 权重缺失时可用：
   ```bash
   cd /root/ColonSplat/submodules/depth_anything
   curl -L https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v1.0.0/depth_anything_vits14.onnx \
     -o weights/depth_anything_vits14.onnx
   ```

5. **onnxruntime 是 CPU 版** —— `pre_dam_dep.py` 会提示 CUDAExecutionProvider 不可用，
   属正常回退，不影响结果（只是慢）。
