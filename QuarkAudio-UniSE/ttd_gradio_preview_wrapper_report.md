# Gradio Preview Wrapper Report

## Basic

- repo_path: `QuarkAudio-UniSE/`
- wrapper_target: `python-function`
- wrapper_mode: `multi-input-preview`
- wrapper_needed: `yes`（官方仓库只有训练/测试脚本，没有产品可用 WebUI）
- preview_validation_level: `real-inference-ok`
- source_runtime_strategy: `docker-preview`
- next_skills: `runtime-containerizer`

## Wrapper Plan

- source_entry: 官方 `model.bicodec.BiCodec`、`model.llm.LLM_SFT` 与官方 Hugging Face 权重
- input_contract: 主音频、SE/TSE/SS 任务选择、TSE 参考人声音频
- output_contract: SE/TSE 返回单声道 WAV；SS 返回两条独立单声道 WAV；统一输出 16 kHz PCM16
- ui_surface_strategy: 单页模式选择；不暴露训练与采样调试参数
- invocation_strategy: `call-python-directly`
- weight_reuse_strategy: 主权重和 WavLM/BiCodec 均复用 `/mnt/TTD-Data/model-previews/quarkaudio-unise/huggingface`
- wrapper_file_plan: `app_preview.py` 负责 UI，`preview_adapter.py` 负责文件输入、分块推理、显存门禁与结果落盘
- launch_strategy: Docker Compose 单卡预览

## Runtime

- wrapper_entry_file: `app_preview.py`
- wrapper_launch_command: `python app_preview.py`
- required_env_vars: `HF_HOME`、`UNISE_OUTPUT_DIR`、`UNISE_MIN_FREE_VRAM_MIB`、`UNISE_MAX_AUDIO_SECONDS`
- required_runtime_dependencies: Torch 2.7.1 CUDA 12.8、Transformers 4.49、Gradio 6.15.1、FFmpeg、libsndfile
- preview_access: `http://ttd-edge:17877/`
- verification_method: 通过 Gradio `/enhance` 分别完成 SE、TSE、SS 真实推理

## Risks

- confirmed_risks: GPU2 同时承载 dots.tts、PYMSS、ClearVoice 等服务；SS 峰值约 6464 MiB，因此提交前要求至少 7500 MiB 空闲显存
- unknowns: 尚未进行多人同时操作的长时间压力测试
- user_decisions_needed: 是否后续把模型测试首页的 ClearVoice 链接切换为统一双 Tab 门户
