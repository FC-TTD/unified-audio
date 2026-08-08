"""TTD preview adapter for QuarkAudio-UniSE.

This module turns the official training/test-oriented implementation into a
small file-in/file-out inference surface. The direct inference flow follows
the public UniSE Space implementation while using the official Alibaba model
source and official Hugging Face checkpoints.
"""

from __future__ import annotations

import gc
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel

from model.bicodec.bicodec import BiCodec
from model.llm import LLM_SFT


SPARK_TTS_REPO = "SparkAudio/Spark-TTS-0.5B"
UNISE_REPO = "QuarkAudio/QuarkAudio-UniSE"
UNISE_CKPT = "epoch=20-step=109367.ckpt"
WAVLM_REPO = "microsoft/wavlm-base-plus"

SAMPLE_RATE = 16_000
SEG_LEN = 5 * SAMPLE_RATE
CHUNK_SECONDS = int(os.environ.get("UNISE_CHUNK_SECONDS", "15"))
CHUNK_LEN = CHUNK_SECONDS * SAMPLE_RATE
MAX_AUDIO_SECONDS = int(os.environ.get("UNISE_MAX_AUDIO_SECONDS", "120"))
MAX_TOTAL_LEN = MAX_AUDIO_SECONDS * SAMPLE_RATE
MIN_FREE_VRAM_MIB = int(os.environ.get("UNISE_MIN_FREE_VRAM_MIB", "7500"))
OUTPUT_DIR = Path(os.environ.get("UNISE_OUTPUT_DIR", "/data/outputs"))
HF_HOME = os.environ.get("HF_HOME", "/models/huggingface")

STFT_CONFIG = {
    "hop_length": 320,
    "win_length": 640,
    "n_fft": 640,
    "n_mels": 80,
}

LLM_CONFIG = {
    "num_tasks": 3,
    "task_map": {"se": 0, "tse": 1, "rtse": 2},
    "feats_dim": 768,
    "llm_base_config": {
        "cond_dim": 80,
        "global_size": 4096,
        "semantic_size": 8192,
        "hidden_size": 512,
        "num_layers": 12,
        "num_attention_heads": 8,
        "dropout_p": 0.1,
        "max_position_embeddings": 4096,
        "label_smoothing": 0.1,
        "conformer_params": {
            "num_layers": 6,
            "dim": 512,
            "heads": 8,
            "dim_head": 64,
            "depthwise_conv_kernel_size": 31,
            "ff_mult": 4,
            "dropout": 0.1,
            "qk_norm": None,
            "pe_attn_head": 1,
        },
    },
}

INFERENCE_LOCK = threading.Lock()


def _decode_audio(audio_path: str) -> tuple[np.ndarray, int]:
    try:
        data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(sample_rate)
    except Exception:
        import librosa

        data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
        if data.ndim > 1:
            data = data.T
        return np.asarray(data, dtype=np.float32), int(sample_rate)


def load_and_resample(audio_path: str) -> torch.Tensor:
    data, sample_rate = _decode_audio(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size == 0:
        raise ValueError("输入音频为空")
    if sample_rate != SAMPLE_RATE:
        data = soxr.resample(data, sample_rate, SAMPLE_RATE, quality="VHQ")
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data)))
    if peak > 1e-6:
        data = data / peak * 0.99
    return torch.from_numpy(np.ascontiguousarray(data)).float()


def gpu_memory() -> tuple[int, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    mib = 1024 * 1024
    return int(free_bytes / mib), int(total_bytes / mib)


class UniSERuntime:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")
        self.bicodec: Optional[BiCodec] = None
        self.semantic_model: Optional[torch.nn.Module] = None
        self.llm: Optional[LLM_SFT] = None
        self._mel_filterbank: Optional[torch.Tensor] = None

    def load(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("容器没有获得预期的单张 CUDA GPU")
        free_mib, total_mib = gpu_memory()
        if free_mib < MIN_FREE_VRAM_MIB:
            raise RuntimeError(
                f"GPU2 当前仅剩 {free_mib} MiB/{total_mib} MiB，"
                f"低于 UniSE 安全门槛 {MIN_FREE_VRAM_MIB} MiB，请稍后重试"
            )

        torch.cuda.reset_peak_memory_stats()
        Path(HF_HOME).mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        bicodec_root = Path(
            snapshot_download(SPARK_TTS_REPO, repo_type="model", cache_dir=HF_HOME)
        )
        self.bicodec = BiCodec.load_from_checkpoint(bicodec_root / "BiCodec")
        self.bicodec = self.bicodec.to(self.device).eval()

        self.semantic_model = AutoModel.from_pretrained(
            WAVLM_REPO,
            cache_dir=HF_HOME,
        ).to(self.device).eval()
        self.semantic_model.requires_grad_(False)

        self.llm = LLM_SFT(**LLM_CONFIG)
        ckpt_path = hf_hub_download(
            UNISE_REPO,
            UNISE_CKPT,
            repo_type="model",
            cache_dir=HF_HOME,
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        clean_state = {
            key[4:] if key.startswith("dnn.") else key: value
            for key, value in state_dict.items()
        }
        missing, unexpected = self.llm.load_state_dict(clean_state, strict=False)
        if missing:
            print(f"UniSE missing keys: {missing}", flush=True)
        if unexpected:
            print(f"UniSE unexpected keys: {unexpected}", flush=True)
        self.llm = self.llm.to(self.device).eval()

    def unload(self) -> None:
        self._mel_filterbank = None
        self.llm = None
        self.semantic_model = None
        self.bicodec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass

    @torch.no_grad()
    def semantic_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        assert self.semantic_model is not None
        waveforms = F.pad(waveforms, (160, 160))
        features = self.semantic_model(waveforms, output_hidden_states=True)
        return torch.stack(features.hidden_states, dim=1).mean(1).detach()

    @torch.no_grad()
    def log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        hop_length = STFT_CONFIG["hop_length"]
        win_length = STFT_CONFIG["win_length"]
        n_fft = STFT_CONFIG["n_fft"]
        n_mels = STFT_CONFIG["n_mels"]
        pad_length = math.ceil(audio.size(-1) / hop_length) * hop_length - audio.size(-1)
        audio = F.pad(
            audio,
            (
                (win_length - hop_length) // 2,
                pad_length + (win_length - hop_length) // 2,
            ),
        )
        spectrum = torch.stft(
            audio,
            n_fft,
            hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length, device=audio.device),
            onesided=True,
            center=False,
            return_complex=True,
        ).transpose(1, 2)
        if self._mel_filterbank is None:
            from torchaudio.functional import melscale_fbanks

            self._mel_filterbank = melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=0.0,
                f_max=8000.0,
                n_mels=n_mels,
                sample_rate=SAMPLE_RATE,
            ).to(audio.device)
        return torch.log(spectrum.abs() @ self._mel_filterbank + 1e-10)

    def _segment(self, chunk: torch.Tensor, normalize: bool) -> torch.Tensor:
        chunk_length = chunk.size(-1)
        pad_length = math.ceil(chunk_length / SEG_LEN) * SEG_LEN - chunk_length
        array = chunk.detach().cpu().numpy()
        if pad_length:
            array = np.pad(array, [(0, 0), (0, pad_length)], mode="wrap")
        segments = torch.from_numpy(array).to(self.device).reshape(-1, SEG_LEN)
        if normalize:
            denominator = chunk.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
            segments = segments / denominator
        return segments

    @torch.no_grad()
    def _decode_tokens(self, global_ids: torch.Tensor, semantic_ids: torch.Tensor) -> torch.Tensor:
        assert self.bicodec is not None
        return self.bicodec.detokenize(semantic_ids, global_ids.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def _run_se(self, chunk: torch.Tensor) -> np.ndarray:
        assert self.llm is not None
        chunk_length = chunk.size(-1)
        segments = self._segment(chunk, normalize=True)
        outputs: list[np.ndarray] = []
        for segment in segments.split(1):
            mix_mel = self.log_mel(segment)
            mix_features = self.semantic_features(segment)
            global_ids, semantic_ids = self.llm.generate(
                task_name="se",
                enroll_mel=None,
                enroll_feats=None,
                mix_mel=mix_mel,
                mix_feats=mix_features,
                do_sample=False,
            )
            estimated = self._decode_tokens(global_ids, semantic_ids)
            outputs.append(estimated.reshape(-1).float().cpu().numpy())
            torch.cuda.empty_cache()
        return np.concatenate(outputs)[:chunk_length]

    @torch.no_grad()
    def _run_tse(
        self,
        chunk: torch.Tensor,
        task_name: str,
        enroll_mel_one: torch.Tensor,
        enroll_features_one: torch.Tensor,
    ) -> np.ndarray:
        assert self.llm is not None
        chunk_length = chunk.size(-1)
        segments = self._segment(chunk, normalize=False)
        outputs: list[np.ndarray] = []
        for segment in segments.split(1):
            mix_mel = self.log_mel(segment)
            mix_features = self.semantic_features(segment)
            global_ids, semantic_ids = self.llm.generate(
                task_name=task_name,
                enroll_mel=enroll_mel_one,
                enroll_feats=enroll_features_one,
                mix_mel=mix_mel,
                mix_feats=mix_features,
                do_sample=False,
            )
            estimated = self._decode_tokens(global_ids, semantic_ids)
            outputs.append(estimated.reshape(-1).float().cpu().numpy())
            torch.cuda.empty_cache()
        return np.concatenate(outputs)[:chunk_length]

    def infer(
        self,
        input_audio: str,
        mode: str,
        enrollment_audio: Optional[str] = None,
    ) -> tuple[str, Optional[str], str]:
        source = load_and_resample(input_audio).to(self.device).unsqueeze(0)
        original_samples = source.size(-1)
        truncated = original_samples > MAX_TOTAL_LEN
        source = source[:, :MAX_TOTAL_LEN]
        total_length = source.size(-1)
        chunk_bounds = [
            (start, min(start + CHUNK_LEN, total_length))
            for start in range(0, total_length, CHUNK_LEN)
        ]

        enrollment = None
        if mode == "tse":
            if not enrollment_audio:
                raise ValueError("目标说话人提取需要上传参考人声")
            enrollment = load_and_resample(enrollment_audio)[:SEG_LEN]
            if enrollment.numel() < SEG_LEN:
                repeats = math.ceil(SEG_LEN / enrollment.numel())
                enrollment = enrollment.repeat(repeats)[:SEG_LEN]
            enrollment = enrollment.to(self.device).unsqueeze(0)

        started = time.time()
        second_output: Optional[str] = None

        if mode == "se":
            parts = [self._run_se(source[:, start:end]) for start, end in chunk_bounds]
            output = np.concatenate(parts)
        elif mode == "tse":
            assert enrollment is not None
            enroll_mel = self.log_mel(enrollment)
            enroll_features = self.semantic_features(enrollment)
            parts = [
                self._run_tse(source[:, start:end], "tse", enroll_mel, enroll_features)
                for start, end in chunk_bounds
            ]
            output = np.concatenate(parts)
        elif mode == "ss":
            head = source[:, :SEG_LEN]
            if head.size(-1) < SEG_LEN:
                repeats = math.ceil(SEG_LEN / head.size(-1))
                head = head.repeat(1, repeats)[:, :SEG_LEN]
            assert self.llm is not None
            global_ids, semantic_ids = self.llm.generate(
                task_name="se",
                enroll_mel=None,
                enroll_feats=None,
                mix_mel=self.log_mel(head),
                mix_feats=self.semantic_features(head),
                do_sample=False,
            )
            enrollment = self._decode_tokens(global_ids, semantic_ids)[:, :SEG_LEN]
            enrollment = enrollment / enrollment.abs().amax().clamp_min(1e-5) * 0.99
            enroll_mel = self.log_mel(enrollment)
            enroll_features = self.semantic_features(enrollment)
            speaker_one = [
                self._run_tse(source[:, start:end], "tse", enroll_mel, enroll_features)
                for start, end in chunk_bounds
            ]
            speaker_two = [
                self._run_tse(source[:, start:end], "rtse", enroll_mel, enroll_features)
                for start, end in chunk_bounds
            ]
            output = np.concatenate(speaker_one)
            second = np.concatenate(speaker_two)
            second_path = OUTPUT_DIR / f"unise_ss_speaker2_{uuid.uuid4().hex}.wav"
            sf.write(second_path, np.clip(second, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
            second_output = str(second_path)
        else:
            raise ValueError(f"不支持的模式: {mode}")

        first_path = OUTPUT_DIR / f"unise_{mode}_{uuid.uuid4().hex}.wav"
        sf.write(first_path, np.clip(output, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
        peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        elapsed = time.time() - started
        warning = f"；输入超过 {MAX_AUDIO_SECONDS} 秒，已截断" if truncated else ""
        status = (
            f"完成：{mode.upper()}，耗时 {elapsed:.1f} 秒，输出 16 kHz PCM16，"
            f"本容器峰值显存 {peak_mib} MiB{warning}"
        )
        return str(first_path), second_output, status


def run_inference(
    input_audio: Optional[str],
    task_mode: str,
    enrollment_audio: Optional[str],
) -> tuple[Optional[str], Optional[str], str]:
    if not input_audio:
        return None, None, "请先上传输入音频"
    mode_map = {
        "语音增强 / 修复（SE）": "se",
        "目标说话人提取（TSE）": "tse",
        "双说话人分离（SS）": "ss",
    }
    mode = mode_map.get(task_mode, "se")
    if not INFERENCE_LOCK.acquire(blocking=False):
        return None, None, "GPU2 正在处理另一个 UniSE 任务，请稍后重试"
    runtime = UniSERuntime()
    try:
        runtime.load()
        return runtime.infer(input_audio, mode, enrollment_audio)
    except torch.OutOfMemoryError:
        return None, None, "GPU2 显存不足，已安全终止本次任务；请等待其他 GPU2 服务释放显存后重试"
    except Exception as exc:
        return None, None, f"处理失败：{type(exc).__name__}: {exc}"
    finally:
        runtime.unload()
        INFERENCE_LOCK.release()


def cuda_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA 不可用"
    free_mib, total_mib = gpu_memory()
    return (
        f"容器 GPU: {torch.cuda.get_device_name(0)}；"
        f"当前空闲 {free_mib}/{total_mib} MiB；"
        f"推理门槛 {MIN_FREE_VRAM_MIB} MiB"
    )
