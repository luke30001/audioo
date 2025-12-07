import base64
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union

import requests
import runpod
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
DEFAULT_CHUNK_LENGTH_S = 30
DEFAULT_MAX_NEW_TOKENS = 448

device = 0 if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def _build_pipeline():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


asr_pipeline = _build_pipeline()


def _decode_base64_to_file(content: str) -> str:
    audio_bytes = base64.b64decode(content)
    handle, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(handle, "wb") as tmp:
        tmp.write(audio_bytes)
    return path


def _download_to_file(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    handle, path = tempfile.mkstemp(suffix=os.path.splitext(url)[1] or ".wav")
    with os.fdopen(handle, "wb") as tmp:
        tmp.write(response.content)
    return path


def _get_audio_file(job_input: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Returns a local audio path and a cleanup path (if a temp file was created).
    """
    cleanup_path: Optional[str] = None
    if "audio_url" in job_input:
        cleanup_path = _download_to_file(job_input["audio_url"])
        return cleanup_path, cleanup_path

    if "audio_base64" in job_input:
        cleanup_path = _decode_base64_to_file(job_input["audio_base64"])
        return cleanup_path, cleanup_path

    if "audio_path" in job_input:
        if not os.path.exists(job_input["audio_path"]):
            raise FileNotFoundError(f"audio_path not found: {job_input['audio_path']}")
        return job_input["audio_path"], None

    raise ValueError("Provide one of: audio_url, audio_base64, or audio_path.")


def _transcribe(
    audio_path: str,
    language: Optional[str],
    timestamps: Union[bool, str],
    chunk_length_s: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    result = asr_pipeline(
        audio_path,
        chunk_length_s=chunk_length_s,
        return_timestamps=timestamps,
        generate_kwargs={
            "language": language,
            "task": "transcribe",
            "max_new_tokens": max_new_tokens,
        },
    )
    return result


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler entrypoint. Expects the standard serverless payload with an
    `input` object as described in the RunPod docs.
    """
    if "input" not in event:
        raise ValueError("Missing required `input` field on event.")

    job_input = event["input"]
    if not isinstance(job_input, dict):
        raise TypeError(f"`input` must be an object, got {type(job_input)!r}.")

    language = job_input.get("language")
    timestamps = job_input.get("timestamps", "word")
    chunk_length_s = float(job_input.get("chunk_length_s", DEFAULT_CHUNK_LENGTH_S))
    max_new_tokens = int(job_input.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))

    audio_path, cleanup_path = _get_audio_file(job_input)

    try:
        output = _transcribe(
            audio_path=audio_path,
            language=language,
            timestamps=timestamps,
            chunk_length_s=chunk_length_s,
            max_new_tokens=max_new_tokens,
        )
    finally:
        if cleanup_path and os.path.exists(cleanup_path):
            os.remove(cleanup_path)

    return output


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
