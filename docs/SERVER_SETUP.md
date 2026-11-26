# EasyCarla-RL server deployment and recovery runbook

本文件按实际实验的顺序整理配置、训练、验证、监控及常见错误修复。旧实验的 GPU 驱动、Python 版本及关键参数见根目录 README。为避免公开云主机的账号信息，SSH 主机与端口使用占位符；数据/权重/完整原始日志保留在自己的服务器，不提交 Git。

## 1. GPU 容器与目录准备

历史环境为 Linux GPU 容器、RTX 4090（nvidia-smi 报告 49140 MiB）、驱动 570.195.03、Python 3.11.8、PyTorch 2.2.1+cu121。`nvidia-smi` 中 CUDA 12.8 是驱动支持版本，不要求 PyTorch 也安装 cu128。需要 GPU 图形/Vulkan 能力，而不仅是 CUDA compute。宿主机与容器设备映射由云平台决定；不要在已经配置好的 GPU 容器中盲目安装或替换驱动。

```bash
nvidia-smi
python --version
df -h /hy-tmp /tmp
free -h
mkdir -p /hy-tmp/easycarla-work
cd /hy-tmp/easycarla-work
```

建议为 CARLA 资源、2.76 GB 数据、训练 checkpoint 与轨迹预留足够磁盘空间；历史盘容量 50 GB，实际使用约 30 GB。容器级内存限制应检查 cgroup，不能仅凭 `free -h` 的宿主机内存断言无内存问题。

在有 root 权限的 Debian/Ubuntu 容器中，可按缺失情况安装通用工具：

```bash
apt-get update
apt-get install -y git wget curl ca-certificates unzip util-linux procps iproute2 \
  libvulkan1 vulkan-tools libglib2.0-0 libgl1 libegl1 libx11-6 libxrandr2 libxi6
```

发行版包名可能不同；这些是重建建议，历史安装的完整 apt 包清单并未保存。NVIDIA 用户态驱动库应来自与宿主机兼容的云镜像/容器运行时，不能以普通 Mesa 替换 NVIDIA ICD。

## 2. CARLA 0.9.16 安装及非 root 用户

从 <https://github.com/carla-simulator/carla/releases/tag/0.9.16> 获取对应 Linux release archive（具体下载文件名由官方发布页确定），放在持久化工作区。CARLA 不是 Python wheel 本身；wheel 只是 RPC 客户端，仿真引擎需要完整 Linux 包。

```bash
cd /hy-tmp/easycarla-work
mkdir -p CARLA_0.9.16
# Replace with the actual downloaded Linux release archive path.
# Inspect archive layout first: tar -tzf /path/to/CARLA_0.9.16.tar.gz | head
# If it contains top-level CarlaUE4.sh, extract directly:
tar -xzf /path/to/CARLA_0.9.16.tar.gz -C CARLA_0.9.16
export CARLA_ROOT=$PWD/CARLA_0.9.16
ls -l "$CARLA_ROOT/CarlaUE4.sh"
find "$CARLA_ROOT/PythonAPI" -name '*cp311*linux*x86_64.whl'
```

UE4 Shipping 不应以 root 启动。原实验使用 `carlasim`（uid/gid 1000）。对于新容器，若没有该账号，建立普通用户并让其拥有 CARLA 目录（仅针对自己的新安装）：

```bash
id carlasim || useradd -m -s /bin/bash carlasim
chown -R carlasim:carlasim "$CARLA_ROOT"
runuser -u carlasim -- test -x "$CARLA_ROOT/CarlaUE4.sh"
runuser -u carlasim -- nvidia-smi
```

## 3. Vulkan offscreen rendering

历史成功配置将 `VK_ICD_FILENAMES`、`VK_DRIVER_FILES` 指向 `/etc/carla-vulkan/nvidia_egl_icd.json`，并为 `carlasim` 创建私有 `XDG_RUNTIME_DIR`（权限 700）。原 ICD JSON 的准确内容没有随归档提供；不要声称下面的重建方式是原文件逐字备份。

先检查现有驱动文件：

```bash
ls -l /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ /etc/carla-vulkan/ 2>/dev/null || true
ldconfig -p | grep -E 'libEGL_nvidia|libGLX_nvidia' || true
```

若原 `/etc/carla-vulkan/nvidia_egl_icd.json` 存在，继续使用并检查可读性。若新容器没有该文件，优先使用云平台安装且验证过的 NVIDIA ICD。只有确认容器内 NVIDIA EGL 库可用于 Vulkan 时，才可尝试以下常见 headless ICD 模板（API version 应匹配驱动，示例并非历史实测 JSON）：

```bash
mkdir -p /etc/carla-vulkan
# Do not overwrite an existing known-working ICD.
if [ ! -e /etc/carla-vulkan/nvidia_egl_icd.json ]; then
cat > /etc/carla-vulkan/nvidia_egl_icd.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0","api_version":"1.3.0"}}
JSON
fi
chmod 644 /etc/carla-vulkan/nvidia_egl_icd.json
runuser -u carlasim -- env \
 VK_ICD_FILENAMES=/etc/carla-vulkan/nvidia_egl_icd.json \
 VK_DRIVER_FILES=/etc/carla-vulkan/nvidia_egl_icd.json \
 vulkaninfo --summary
```

若 Vulkan 检查不能枚举 NVIDIA GPU，先处理设备映射和驱动库；不要反复启动仿真器。CARLA RPC 默认端口 2000、相关 2001/2002，只用于自己的可信环境，不应直接暴露公网。

## 4. Python, EasyCarla and dataset

以本仓库根目录为起点，Python 3.11 venv/GPU 容器中执行：

```bash
export CARLA_ROOT=/hy-tmp/easycarla-work/CARLA_0.9.16
bash scripts/install_python.sh
bash scripts/install_workspace.sh /hy-tmp/easycarla-work/USPI-EasyCarla-RL
cd /hy-tmp/easycarla-work/USPI-EasyCarla-RL
python -c 'import carla,gym,numpy,torch,easycarla; print(torch.__version__, torch.cuda.is_available())'
mkdir -p data logs da_runs demo_runs
wget -c 'https://huggingface.co/datasets/silverwingsbot/easycarla/resolve/main/easycarla_offline_dataset.hdf5' -O data/easycarla_offline_dataset.hdf5
sha256sum data/easycarla_offline_dataset.hdf5
```

实际历史安装使用 CARLA 自带 cp311 wheel、`pip install -e . --no-deps --no-build-isolation`，避免 upstream requirements 强制降级 PyTorch。原 upstream commit 未保存；`install_workspace.sh` 会记录当前 clone commit 并覆盖为归档中实际使用的 agent/env 文件。若原工作区仍在，可执行 `git rev-parse HEAD` 记录历史 commit，再用 `EASYCARLA_REF` 明确指定。缺少完整原始依赖锁、CARLA 发行包 hash 与数据 hash，故这是可复现流程，不是已证实 bitwise reproduction。

数据下载完成后，校验 HDF5 结构和长度（不启动 CARLA 或训练）：

```bash
python - <<'PY'
import h5py
with h5py.File('data/easycarla_offline_dataset.hdf5','r') as f:
    for k in ['observations','actions','next_observations','rewards','costs','done','info/is_collision','info/is_off_road']:
        print(k, f[k].shape, f[k].dtype)
    assert f['observations'].shape[1]==307 and f['actions'].shape[1]==3
PY
```

历史数据在同一文件中按完整轨迹分割：`split_seed=2026`, loader horizon=10 先进行 RNG 邻接抽样检查，随后分配 train/cal/test。不要随意改成 horizon=1，否则即便 seed 一样也可能改变分割。数据加载器需要将当前/下一观测读入 CPU RAM。

## 5. Startup, functional validation and Town03

在确认没有其他仿真引擎或控制程序运行时，执行：

```bash
pgrep -af '[C]arlaUE4' || true
ss -lntp | grep -E ':2000|:2001|:2002' || true
nohup runuser -u carlasim -- env CARLA_ROOT="$CARLA_ROOT" \
  CARLA_VULKAN_ICD=/etc/carla-vulkan/nvidia_egl_icd.json \
  bash "$PWD/scripts/start_carla.sh" > logs/carla_server.log 2>&1 < /dev/null &
echo $! > logs/carla_server.pid
tail -F logs/carla_server.log
```

`Ctrl+C` 只退出 tail。另一个服务器终端运行 `python scripts/check_carla.py`。正常状态需要 RPC 版本匹配、世界可获取、异步时钟持续推进，而不只是监听端口。原部署曾在进程存在、三个端口均监听、GPU 分配约4.5 GB 的情况下持续出现 `Waited for 10 seconds on IO` 与 RPC timeout。加入 `-NoAsyncLoadingThread` 后用户确认服务成功响应，默认地图 Town10HD_Opt、155 个 spawn points，时钟连续推进。该诊断参数与恢复存在时间关联，但不能证明唯一根因。

切换到 Town03：

```bash
# MUST stop every evaluation / demo first; the following reload is global.
python scripts/prepare_town03.py --confirm-idle
```

脚本先解除同步模式与固定步长、关闭渲染，加载 Town03，检查10帧，开启渲染，再检查10帧。历史 Town03 有265个 spawn points。实际评估器随后配置自己的同步模式与 0.1 s 固定步长。已有 CARLA 世界处于同步模式时，健康检查不会强行 tick，避免破坏评估完整性。

## 6. Offline training and evaluation (optional reproduction)

本次归档不启动任何新实验。训练完全离线，不依赖 CARLA。如果已有 checkpoint，跳过训练。请保持原 `da_runs/rcdpi42` 不变，使用新的输出目录。DQL 训练命令与 FQE 完整参数已在 README 和 `scripts/train_offline.sh` 中记录。

历史 RCDPI V1 训练先完成 3 × 66,667 成员更新，之后依次运行 RCDPI / Rule / DQL，各100回合。原评估脚本使用 `fcntl` advisory lock、地图复用检查、单帧推进检查及控制回读，不会自行重载地图。两个并行控制器可能导致全局车辆/传感器被清理；不要同时运行 demo 与 evaluation。早期 pipeline 在 CARLA 未启动时提前失败；改进后的做法是先离线训练与保存完整模型，再连接 CARLA。若 CARLA 超时，保留权重，恢复服务后只执行尚未完成的评估；不重复训练。

```bash
ACTOR=da_runs/strong42_eta1/dql_epoch0200.pth
CRITICS=da_runs/rcdpi42/critics.pt
# Historical V1 reproduction, fresh output directory only:
python -u evaluate_da_control.py --mode da_control --checkpoint "$ACTOR" \
  --control-checkpoint "$CRITICS" --epsilon-cost 0 \
  --out da_runs/rcdpi_reproduction/eval_rcdpi --episodes 100 --seed 1000 \
  --vehicles 20 --steps 1000 --town Town03 --device cuda
```

Rule and DQL use `--mode rule` or `--mode dql --checkpoint "$ACTOR"`, with separate fresh outputs and the same seeds. USPI evaluation entrypoint/optional three-policy script is in README. If evaluations fail, preserve `INVALID.txt` and partial outputs for diagnosis; do not append to, overwrite or pass off partial episodes as a verified cohort.

## 7. Dashboard, rendering and operational checks

历史 V5 双视角由 8080 内部 MJPEG 服务提供，8081 只读监控负责代理视频、显示曲线与同 seed 的三算法轨迹。README 给出启动命令。离线训练或正式评估不产生实时视频；显示黑色占位不意味着车辆没有运行。Demo 采用独立目录与同一评估器推理机制，默认10个回合。不要为了无限循环将回合数设置极大而不监控磁盘。第三个视图是 XY 轨迹回放，非 CARLA 三车视频。未提供历史原始 CSV 时页面仅显示缺失，不虚构曲线。

Windows PowerShell 或本地终端：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
  -L 8081:127.0.0.1:8081 -p <SSH_PORT> <USER>@<SSH_HOST>
```

访问 `http://127.0.0.1:8081/`。SSH tunnel 不影响服务器自身 `127.0.0.1:2000` CARLA RPC。Dashboard 健康检查用 GET：

```bash
curl -sS -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8081/
curl -fsS http://127.0.0.1:8081/api/comparison?seed=1000 | python -m json.tool
curl -fsS http://127.0.0.1:8080/stats
curl -fsS http://127.0.0.1:8080/dual_camera_status
```

`curl -I` 会发送 HEAD，历史 HTTP handler 可能返回501；应改用 GET。8081 监听本地回环即可，不需要公开 8080。

验证监控（历史实验路径）：

```bash
watch -n 3 'for n in rcdpi rule dql; do f="da_runs/rcdpi42/eval_$n/episodes.csv"; if [ -s "$f" ]; then printf "%s: %s/100\n" "$n" "$(($(wc -l < "$f")-1))"; else echo "$n: 0/100"; fi; done; tail -n 8 logs/rcdpi42_pipeline.log; nvidia-smi'
```

## 8. Troubleshooting checklist

| Symptom | Check / remedy |
|---|---|
| `No such file or directory` for startup shell | CARLA binary may still exist; recreate shell from `scripts/start_carla.sh`, do not reinstall models |
| `Exit 127` | Verify executable path and permissions before restarting |
| No process or 2000 listener | Check server launch log, user and ICD |
| Listening but RPC timeout | Check engine IO/load, process and GPU; test RPC version + world + ticks; engine readiness is not just TCP readiness |
| `Waited for 10 seconds on IO` | Check storage and thread states; only when no clients, controlled restart using `-NoAsyncLoadingThread` may help |
| `/proc/<pid>/io` denied in container | Try same CARLA user; do not weaken host security or infer actual IO data from error |
| Town10HD_Opt after boot | Use exclusive Town03 preparation with renderer temporarily off |
| Missing initial LiDAR | Check rendering, sensor initialization, sync mode and no other controller; original evaluator waits up to15 s |
| 8081 listens but local browser fails | Server GET check first, then SSH tunnel; no need to restart training |
| No live video | Check demo 8080 `/stats`, `/dual_camera_status`; do not start another controller during validation |
| Reuse trained checkpoint | Use trusted local `.pth`/`.pt`, verify hashes if available, skip offline training |

## 9. Archive and publish safely

The repository ignores model checkpoints, datasets, archives, logs, credentials and server outputs by default. Do not run `git add .` from the entire original EasyCarla/CARLA workspace. The archive prepared for publication contains only source, configuration, documentation and limited historical aggregate results. Before uploading additional result files, inspect for host addresses, tokens, passwords, personal data and data-license restrictions. To reproduce historical results precisely, preserve the local original data/checkpoints/configs and hash them separately.

A GitHub account with repository creation and push permissions is required. Use `gh auth login` locally; never paste an access token into the README, command history or chat. Use `bash scripts/publish_github.sh` from the extracted repository. It defaults to a private new repository; set `VISIBILITY=public` only if you want the code public. If the repository exists, the script refuses to overwrite it. No force pushes.
