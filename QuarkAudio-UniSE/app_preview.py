"""Minimal Gradio preview for QuarkAudio-UniSE on TTD Edge."""

import os

import gradio as gr

from preview_adapter import MAX_AUDIO_SECONDS, cuda_summary, run_inference


CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto; }
.risk-note { border-left: 4px solid #d97706; padding-left: 12px; }
"""

with gr.Blocks(title="QuarkAudio-UniSE 生成式语音修复") as demo:
    gr.Markdown("# QuarkAudio-UniSE｜生成式语音修复")
    gr.Markdown(
        "基于 Decoder-only 自回归音频语言模型，通过 Codec Token 重构语音。"
        "适合严重噪声、混响、丢包、剪切和信息缺失场景。"
    )
    gr.Markdown(
        "<div class='risk-note'><b>保真风险：</b>极低信噪比下可能改变细小发音、音色或语气。"
        "司法、证据性或要求逐字忠实的素材请优先使用 ClearVoice，并人工复核。</div>"
    )
    gr.Markdown(
        f"当前预览最长处理 **{MAX_AUDIO_SECONDS} 秒**；输出统一为 **16 kHz PCM16 WAV**。"
        "UniSE 的 Speech Restoration 不等同于 ClearVoice 的 48 kHz Speech Super-Resolution。"
    )
    gpu_status = gr.Markdown(
        f"GPU 状态（页面启动时，仅供参考；提交任务会再次检查）：{cuda_summary()}"
    )

    with gr.Row():
        with gr.Column():
            task_mode = gr.Radio(
                [
                    "语音增强 / 修复（SE）",
                    "目标说话人提取（TSE）",
                    "双说话人分离（SS）",
                ],
                value="语音增强 / 修复（SE）",
                label="任务",
            )
            input_audio = gr.Audio(
                label="输入音频（噪声、损坏或多人混合语音）",
                type="filepath",
            )
            enrollment_audio = gr.Audio(
                label="目标说话人参考音频（仅 TSE）",
                type="filepath",
                visible=False,
            )
            run_button = gr.Button("开始生成式修复", variant="primary")
        with gr.Column():
            output_one = gr.Audio(label="输出 / 说话人 1", type="filepath")
            output_two = gr.Audio(label="说话人 2", type="filepath", visible=False)
            status = gr.Markdown("等待任务")

    def update_mode(mode: str):
        return (
            gr.update(visible="TSE" in mode),
            gr.update(visible="SS" in mode),
        )

    task_mode.change(
        update_mode,
        inputs=task_mode,
        outputs=[enrollment_audio, output_two],
    )
    run_button.click(
        run_inference,
        inputs=[input_audio, task_mode, enrollment_audio],
        outputs=[output_one, output_two, status],
        api_name="enhance",
    )

demo.queue(default_concurrency_limit=1, max_size=4)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        css=CSS,
        allowed_paths=[os.environ.get("UNISE_OUTPUT_DIR", "/data/outputs")],
    )
