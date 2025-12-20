# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import copy
import torch
import time
import subprocess
import json
from typing import Optional, List
from cog import BasePredictor, Input, Path, BaseModel
from peft import PeftModel

# Set up model cache
MODEL_CACHE = "model_cache"

# Critical: Set CUDA_HOME for DeepSpeed
os.environ["CUDA_HOME"] = "/usr/local/cuda"

# Add local llava module to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import llava

BASE_URL = "https://weights.replicate.delivery/default/audio-flamingo-3/model_cache/"

def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class ModelOutput(BaseModel):
    """Output schema for the model prediction"""
    response: str
    embeddings_json: str
    logs: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Create model cache directory if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # Set environment variables for model caching
        os.environ["HF_HOME"] = MODEL_CACHE
        os.environ["TORCH_HOME"] = MODEL_CACHE
        os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
        os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
        os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

        model_files = [
            ".locks.tar",
            "models--Qwen--Qwen2-Audio-7B.tar",
            "models--nvidia--audio-flamingo-3-chat.tar",
            "models--nvidia--audio-flamingo-3.tar",
            "version.txt",
        ]

        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Set up model paths
        self.MODEL_BASE_SINGLE = os.path.join(MODEL_CACHE, "models--nvidia--audio-flamingo-3", "snapshots", "504150e751238e1471971f8bef3303e32b5fd23d")
        self.MODEL_BASE_THINK = os.path.join(self.MODEL_BASE_SINGLE, 'stage35')
        print(f"[+] Model paths set: {self.MODEL_BASE_SINGLE}")

        print("[+] Loading single-turn model...")
        self.model_single = llava.load(self.MODEL_BASE_SINGLE, model_base=None)
        self.model_single = self.model_single.to("cuda")
        self.model_single_copy = copy.deepcopy(self.model_single)
        self.generation_config_single = self.model_single.default_generation_config
        print("[+] Single-turn model loaded successfully")

        print("[+] Loading think model...")
        self.model_think = PeftModel.from_pretrained(
            self.model_single,
            self.MODEL_BASE_THINK,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("[+] Think model loaded successfully")
        print("[+] Model setup complete!")

    def extract_audio_embeddings(self, audio_path: str) -> str:
        """
        Extract audio embeddings - SIMPLE VERSION THAT WORKS

        Returns JSON with either {"vector": [...]} or {"error": "..."}
        """
        try:
            from llava.utils.media import extract_media
            from llava.mm_utils import process_sounds, process_sound_masks

            # Create sound object and extract
            sound = llava.Sound(audio_path)
            conversation = [{"from": "human", "value": [sound]}]
            media, media_meta = extract_media(conversation, self.model_single.config)

            # Process
            sounds = process_sounds(media["sound"]).half().cuda()
            masks = process_sound_masks(media_meta["sound_feature_masks"]).half().cuda()

            with torch.no_grad():
                # Get encoder output (before mm_projector)
                sound_tower = self.model_single.get_sound_tower()
                features = sound_tower(sounds, masks)

                # If list, stack
                if isinstance(features, list):
                    features = torch.stack(features, dim=0)

                # Force to 1D by pooling all dims except last, then flatten
                while features.ndim > 1:
                    features = features.mean(dim=0)

                # Ensure 1D
                features = features.squeeze().flatten()

                # Convert to list
                vec = features.cpu().float().tolist()

                # Validate
                if not isinstance(vec, list):
                    return json.dumps({"error": f"Not a list: {type(vec)}"})

                if len(vec) == 0:
                    return json.dumps({"error": "Empty vector"})

                if len(vec) > 10000:
                    return json.dumps({"error": f"Too large: {len(vec)} dims"})

                # Check first element
                if isinstance(vec[0], (list, tuple)):
                    return json.dumps({"error": "Nested structure detected"})

                # Success
                return json.dumps({"vector": vec})

        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "trace": traceback.format_exc()})

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to analyze. Supports speech, music, and sound effects. Maximum duration: 10 minutes."
        ),
        prompt: str = Input(
            description="Question or instruction about the audio",
            default="Please describe this audio in detail."
        ),
        system_prompt: str = Input(
            description="System instructions to customize the model's behavior, output format, or analysis style. Leave empty for default behavior.",
            default=""
        ),
        enable_thinking: bool = Input(
            description="Enable detailed chain-of-thought reasoning for complex analysis. False for faster responses, True for deeper insights.",
            default=False
        ),
        temperature: float = Input(
            description="Controls response creativity and randomness. Use 0.0 for deterministic (default), 0.1-0.3 for factual analysis, 0.7-0.9 for creative interpretation.",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        max_length: int = Input(
            description="Maximum length of the response in tokens. Use 0 for model default, or specify 50-2048 for custom length.",
            default=0,
            ge=0,
            le=2048
        ),
        start_time: Optional[float] = Input(
            description="Start time in seconds for audio segment analysis (optional). Useful for long audio files.",
            default=None
        ),
        end_time: Optional[float] = Input(
            description="End time in seconds for audio segment analysis (optional). Must be greater than start_time.",
            default=None
        ),
        embeddings_only: bool = Input(
            description="If True, only return the audio embeddings without generating a text response. Useful for embedding-based tasks.",
            default=False
        ),
    ) -> ModelOutput:
        """Analyze audio using Audio Flamingo 3 - supports speech, music, and sound analysis up to 10 minutes"""

        # Validate audio segment timing
        if start_time is not None and start_time < 0:
            raise ValueError("start_time must be non-negative")
        if end_time is not None and end_time < 0:
            raise ValueError("end_time must be non-negative")
        if start_time is not None and end_time is not None:
            if end_time <= start_time:
                raise ValueError("end_time must be greater than start_time")

        # Extract embeddings
        embeddings_json = self.extract_audio_embeddings(str(audio))

        # Parse to add to logs
        logs_lines = []
        try:
            parsed = json.loads(embeddings_json)
            if "vector" in parsed:
                logs_lines.append(f"Embeddings extracted: {len(parsed['vector'])} dimensions")
            elif "error" in parsed:
                logs_lines.append(f"Embedding extraction error: {parsed['error']}")
        except:
            logs_lines.append("Could not parse embeddings_json")

        # If embeddings_only mode, return early
        if embeddings_only:
            return ModelOutput(
                response="[embeddings_only mode - no text response generated]",
                embeddings_json=embeddings_json,
                logs="\n".join(logs_lines)
            )

        # Create sound object with optional segment timing
        if start_time is not None or end_time is not None:
            import librosa
            import soundfile as sf
            import tempfile

            y, sr = librosa.load(str(audio))
            start_sample = int(start_time * sr) if start_time is not None else 0
            end_sample = int(end_time * sr) if end_time is not None else len(y)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, y[start_sample:end_sample], sr)
                sound = llava.Sound(tmp_file.name)
        else:
            sound = llava.Sound(str(audio))

        # Prepare generation config
        generation_config = copy.deepcopy(self.generation_config_single)

        if max_length > 0:
            generation_config.max_new_tokens = max_length

        if temperature > 0:
            generation_config.temperature = temperature
            generation_config.do_sample = True

        # Construct prompt
        if system_prompt.strip():
            full_prompt = f"<sound>\n{system_prompt.strip()}\n\n{prompt}"
        else:
            full_prompt = f"<sound>\n{prompt}"

        if enable_thinking:
            response = self.model_think.generate_content(
                [sound, full_prompt],
                generation_config=generation_config
            )
        else:
            response = self.model_single_copy.generate_content(
                [sound, full_prompt],
                generation_config=generation_config
            )

        logs_lines.append(f"Response generated: {len(response)} characters")

        return ModelOutput(
            response=response,
            embeddings_json=embeddings_json,
            logs="\n".join(logs_lines)
        )
