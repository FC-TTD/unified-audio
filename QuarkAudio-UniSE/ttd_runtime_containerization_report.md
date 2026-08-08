# Runtime Containerization Report

## Basic

- repo_path: `QuarkAudio-UniSE/`
- target_mode: `preview`
- runtime_strategy: `docker-preview`
- source_entry: `python app_preview.py`
- source_ports: `7860`（宿主映射 `17877`）
- source_artifacts: `/data/outputs`

## Container Plan

- build_context_strategy: 项目子目录作为 context；`.dockerignore` 排除 Git、训练集、权重、输出、缓存和非运行资料
- base_image_strategy: 复用 ttd-edge 已缓存的 `pymss-studio-preview:latest`，提供 Torch 2.7.1 + CUDA 12.8 + tini
- python_runtime_strategy: 基础镜像 Conda Python 3.11
- dependency_install_strategy: `pip install -r requirements.preview.txt`
- gpu_strategy: `ttd-edge / physical GPU2`；Compose `device_ids: ["2"]` 与 `NVIDIA_VISIBLE_DEVICES=2`；容器内仅见一张 RTX 3090
- weight_strategy: 不 bake 权重；目标机首次预热直接下载至共享存储
- project_weight_mounts: `[]`
- runtime_model_cache_mounts: `/mnt/TTD-Data/model-previews/quarkaudio-unise/huggingface:/models/huggingface`
- cache_mounts: 同上
- data_mounts: `/opt/quarkaudio-unise-preview/outputs:/data/outputs`
- first_run_warmup_strategy: 首次真实 SE 请求预下载 UniSE、WavLM、BiCodec；缓存实际约 5.3 GiB
- env_file_strategy: Compose 显式环境变量，无 secret
- system_packages: `ffmpeg`、`libsndfile1`
- reuse_upstream_assets: 复用官方模型源码；新增最小 preview Docker/Gradio 资产
- private_index_strategy: 不使用私有 PyPI
- build_cache_strategy: BuildKit apt/pip cache；依赖层位于高频应用源码之前
- network_strategy: 默认 Compose bridge；宿主端口直出，不注册稳定 Caddy 名称
- volume_strategy: 共享模型缓存使用宿主绝对路径，输出使用项目目录 bind mount
- proxy_strategy: 统一门户使用独立 nginx iframe 聚合，不改变 ClearVoice 服务

## Runtime

- image_build_command: `docker build --network host -f Dockerfile.preview -t quarkaudio-unise-preview:edge .`
- run_command: `docker compose -f compose.preview.yaml up -d`
- compose_project_name: `quarkaudio-unise-preview`
- container_naming_strategy: 为预览运维固定 `quarkaudio-unise-preview` 与 `speech-restoration-lab-preview`
- preview_compose_filename: `compose.preview.yaml`
- preview_deploy_script_filename: `deploy_preview.sh`
- compose_needed: `yes`
- stack_needed: `no`
- deploy_topology: `local-compose` on remote host `ttd-edge`
- healthcheck_strategy: Gradio 首页 HTTP 探活；nginx 门户首页探活
- verification_method: 容器内 CUDA 冒烟、HTTP 首页、Gradio API 真实生成、重启后再次生成
- expected_access: UniSE `http://ttd-edge:17877/`；统一门户 `http://ttd-edge:17878/`

## Risks

- confirmed_risks: 本地系统盘使用率约 88%；镜像约 8.87 GB；GPU2 当前可用显存接近 7.5 GB 门槛
- unknowns: 基础镜像是 TTD 本地预览资产，尚未发布到独立 registry
- user_decisions_needed: 是否后续转正式仓库、固定镜像摘要并接入内部 Caddy
