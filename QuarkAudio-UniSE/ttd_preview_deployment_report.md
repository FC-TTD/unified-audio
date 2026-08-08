# QuarkAudio-UniSE 与 ClearVoice 双 Tab 预览部署报告

## 服务与意图

- lane: `server-preview`
- target: `ttd-edge / physical GPU2`
- official_source: `alibaba/unified-audio@c4004e217ddf7c514f72ea22f2d0fbf43f02ae90`
- license: Apache-2.0
- local_branch: `ttd`（未 commit、未 push）
- deployment_root: `/opt/quarkaudio-unise-preview`
- deployment_asset: `/opt/quarkaudio-unise-preview/compose.preview.yaml`
- unified_portal: `http://ttd-edge:17878/`
- navigation_entry: `testing.proc.clearervoice` / `降噪与人声提取`
- unise_direct: `http://ttd-edge:17877/`
- existing_clearvoice: `http://ttd-edge:17875/`

## 契约发现

- UniSE 主能力: SE 语音增强/恢复、TSE 目标说话人提取、SS 双说话人分离
- UniSE 输入: 主音频；TSE 额外需要参考人声
- UniSE 输出: 16 kHz PCM16 WAV；SS 输出两条独立音轨
- ClearVoice 保持现有 Gradio 接口与容器，不修改、不停止
- 统一门户: 两个 Tab 分别内嵌 ClearVoice 与 UniSE，并提供单独打开链接

## Placement Decision

- selected: `ttd-edge GPU2`，由用户明确确认
- baseline: GPU2 审计时约占用 15.7/24 GiB，主要来自 dots.tts（约 12 GiB）和 PYMSS（约 3 GiB）
- mitigation: UniSE 请求时加载、请求后卸载；单并发；提交前至少要求 7500 MiB 空闲显存
- exception: 不停止或替换任何现有 GPU2 服务

## Desired / Management / Runtime

- desired: 本地 `ttd` 分支上的 preview 文件同步到 `/opt/quarkaudio-unise-preview`
- management: Compose project `quarkaudio-unise-preview`
- runtime: `quarkaudio-unise-preview` 与 `speech-restoration-lab-preview` 均 running/healthy
- deployed_image: `sha256:370500e6e047d2373971ec68004a18ebc5f96b26b08b841078e9658dde848556`
- previous_image: N/A（新服务）
- key_asset_hashes:
  - `Dockerfile.preview`: `7e56559245bd0f33ae0e0157b5927f179607d020d8ff21b432f7a4ec5a20d008`
  - `compose.preview.yaml`: `24a1b16fd2f1a21325bf119f386686e60d86b9fd9a62ad45185fab21676a1ddf`
  - `app_preview.py`: `c30d5442ffdf8951091fc00b58222bba8e8f36cda0cfe2f09a5acb718f27b498`
  - `preview_adapter.py`: `ddf626aa9191c5dd96ddfd5109e3fcad978b9f4eb6814a84400a2e09ad0cb74d`
  - `portal/index.html`: `d21f5df5c270864a3a2dafd3eb1a1decb90044b4c6aa674f8b6e1c635fd400f9`

## 执行动作

1. 审计 GPU2、容器、端口、共享存储与 ClearVoice 运行状态。
2. 从官方仓库抽取轻量推理入口，补 Transformers 兼容层。
3. 构建 Gradio preview、Dockerfile、Compose、持久化缓存与输出目录。
4. 在 `17877` 启动 UniSE，在 `17878` 启动双 Tab nginx 门户。
5. 预热官方 UniSE、WavLM、BiCodec 权重到共享缓存。
6. 完成 SE/TSE/SS、ClearVoice 回归、容器重启和浏览器渲染验收。

## 基础存活证据

- `http://ttd-edge:17877/`: HTTP 200
- `http://ttd-edge:17878/`: HTTP 200
- UniSE 容器: healthy
- 门户容器: healthy
- 容器内 CUDA: available；device_count=1；RTX 3090；Torch 2.7.1 + CUDA 12.8
- GPU 绑定: `NVIDIA_VISIBLE_DEVICES=2` 且 DeviceRequest `device_ids=["2"]`

## 真实业务 Smoke

### UniSE SE

- input: 2.398 秒、16 kHz、单声道中文语音加白噪声
- result: 成功，约 3.0-3.1 秒
- output: 16 kHz PCM16 单声道 WAV，时长 2.398 秒
- peak_vram: 约 1502 MiB
- restart_check: 容器 restart/recreate 后再次成功

### UniSE TSE

- input: 官方社区示例混合语音 + 目标说话人参考音频
- result: 成功，约 2.3 秒
- output: 16 kHz PCM16 单声道 WAV，时长 4.905 秒
- peak_vram: 约 1504 MiB

### UniSE SS

- input: 官方社区双说话人示例
- result: 成功，约 8.5 秒
- output: 两条 16 kHz PCM16 单声道 WAV，时长均为 4.739 秒
- peak_vram: 约 6464 MiB

### ClearVoice 回归

- input: 现有 `mandarin_speech_16kHz.wav`
- model: `FRCRN_16000Hz`
- result: 成功
- output: 16 kHz PCM16 单声道 WAV，时长 2.398 秒

## UI 验收

- 桌面视口: ClearVoice 与 UniSE iframe 均真实加载；Tab 切换状态正确
- 移动视口 390x844: 两个 Tab 可见，UniSE 表单可滚动使用，无横向溢出阻塞
- 浏览器控制台: 无相关 error/warn
- 风险文案: 已区分生成式修复与保真降噪，并说明 UniSE Speech Restoration 不等同于 ClearVoice 48 kHz Super-Resolution

## 风险、未知项与回退

- GPU2 推理后空闲显存约 7650 MiB，接近 7500 MiB 门槛；其他服务繁忙时 UniSE 会拒绝任务并提示稍后重试。
- 当前最长输入限制为 120 秒，避免共享 GPU 上的超长自回归任务。
- 模型缓存约 5.3 GiB，位于共享存储；不删除、不覆盖现有模型缓存。
- 模型测试首页原 ClearVoice 条目已保留稳定 key `testing.proc.clearervoice`，显示名改为“降噪与人声提取”，并切换到双模型门户。
- rollback: 如后续明确授权停止，可在 `/opt/quarkaudio-unise-preview` 仅停止新建 Compose project；现有 ClearVoice 不受影响。

## Verdict

`success`：指定 GPU2 部署、双 Tab 页面、三条 UniSE 真实推理、ClearVoice 回归和重启持久性均通过。
