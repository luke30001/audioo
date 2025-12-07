"""
Entry point for RunPod Serverless.

This imports the transcription handler and starts the RunPod worker runtime.
"""

import runpod

from handler import handler


runpod.serverless.start({"handler": handler})
