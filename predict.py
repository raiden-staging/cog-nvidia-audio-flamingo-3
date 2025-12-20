# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import copy
import torch
import time
import subprocess
from typing import List, Optional, Any
from cog import BasePredictor, Input, BaseModel
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


class AudioResult(BaseModel):
    """Result for a single audio file"""
    index: int
    embeddings: Optional[List[float]] = None
    embeddings_error: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None


class ModelOutput(BaseModel):
    """
    Output schema - array of results

    results: Array of AudioResult objects, one per input audio file
    logs: Processing logs
    """
    results: List[AudioResult]
    logs: str


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

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

    def extract_audio_embeddings(self, audio_path: str) -> tuple:
        """
        Extract audio embeddings from the encoder.

        The dimensionality (1280) comes from AF-Whisper encoder's hidden_size
        in the model architecture - NOT hardcoded.

        Returns: (embeddings: List[float] or None, error: str or None)
        """
        try:
            from llava.utils.media import extract_media
            from llava.mm_utils import process_sounds, process_sound_masks

            sound = llava.Sound(audio_path)
            conversation = [{"from": "human", "value": [sound]}]
            media, media_meta = extract_media(conversation, self.model_single.config)

            sounds = process_sounds(media["sound"]).half().cuda()
            masks = process_sound_masks(media_meta["sound_feature_masks"]).half().cuda()

            with torch.no_grad():
                # Get encoder output (shape: [..., hidden_size] where hidden_size=1280 from architecture)
                sound_tower = self.model_single.get_sound_tower()
                features = sound_tower(sounds, masks)

                # If list, stack
                if isinstance(features, list):
                    features = torch.stack(features, dim=0)

                # Force to 1D: pool all dims except last, then flatten
                while features.ndim > 1:
                    features = features.mean(dim=0)

                features = features.squeeze().flatten()

                # Convert to Python list
                vec = features.cpu().float().tolist()

                # Validate
                if not isinstance(vec, list):
                    return None, f"Not a list: {type(vec)}"
                if len(vec) == 0:
                    return None, "Empty vector"
                if len(vec) > 10000:
                    return None, f"Too large: {len(vec)} dims"
                if len(vec) > 0 and isinstance(vec[0], (list, tuple)):
                    return None, "Nested structure detected"

                return vec, None

        except Exception as e:
            import traceback
            return None, f"{str(e)}\n{traceback.format_exc()}"

    def download_audio_from_url(self, url: str) -> str:
        """Download audio from URL to temp file, return path"""
        import tempfile
        import urllib.request

        ext = ".wav"
        if url.lower().endswith(".mp3"):
            ext = ".mp3"
        elif url.lower().endswith(".m4a"):
            ext = ".m4a"
        elif url.lower().endswith(".flac"):
            ext = ".flac"

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name
        temp_file.close()

        urllib.request.urlretrieve(url, temp_path)
        return temp_path

    def process_single_audio(
        self,
        audio_path: str,
        prompt: str,
        system_prompt: str,
        enable_thinking: bool,
        temperature: float,
        max_length: int,
        embeddings_only: bool,
        item_index: int
    ) -> AudioResult:
        """
        Process one audio file, return AudioResult.
        """
        result = AudioResult(
            index=item_index,
            embeddings=None,
            embeddings_error=None,
            response=None,
            error=None
        )

        try:
            # ALWAYS extract embeddings
            embeddings, emb_error = self.extract_audio_embeddings(audio_path)

            if emb_error:
                result.embeddings_error = emb_error
            else:
                result.embeddings = embeddings

            # If embeddings_only, done
            if embeddings_only:
                result.response = "[embeddings_only mode]"
                return result

            # Generate text response
            sound = llava.Sound(audio_path)
            generation_config = copy.deepcopy(self.generation_config_single)

            if max_length > 0:
                generation_config.max_new_tokens = max_length

            if temperature > 0:
                generation_config.temperature = temperature
                generation_config.do_sample = True

            if system_prompt.strip():
                full_prompt = f"<sound>\n{system_prompt.strip()}\n\n{prompt}"
            else:
                full_prompt = f"<sound>\n{prompt}"

            if enable_thinking:
                response = self.model_think.generate_content([sound, full_prompt], generation_config=generation_config)
            else:
                response = self.model_single_copy.generate_content([sound, full_prompt], generation_config=generation_config)

            result.response = response
            return result

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            result.error = error_msg
            result.response = "[error during processing]"
            if not result.embeddings_error:
                result.embeddings_error = str(e)
            return result

    def predict(
        self,
        audio_files: List[str] = Input(
            description='List of audio URLs or paths. Example: ["https://example.com/audio1.mp3", "https://example.com/audio2.wav"]'
        ),
        prompt: str = Input(
            description="Question or instruction about the audio. Applied to all audio files.",
            default="Please describe this audio in detail."
        ),
        system_prompt: str = Input(
            description="System instructions to customize behavior. Leave empty for default.",
            default=""
        ),
        enable_thinking: bool = Input(
            description="Enable chain-of-thought reasoning. False=faster, True=deeper analysis.",
            default=False
        ),
        temperature: float = Input(
            description="Response randomness. 0.0=deterministic, 0.7-0.9=creative.",
            default=0.0,
            ge=0.0,
            le=1.0
        ),
        max_length: int = Input(
            description="Max response tokens. 0=model default, or specify 50-2048.",
            default=0,
            ge=0,
            le=2048
        ),
        embeddings_only: bool = Input(
            description="If true, only return embeddings without text. Applies to all files.",
            default=False
        ),
    ) -> ModelOutput:
        """
        Process audio files with Audio Flamingo 3.

        Input: List of audio URLs/paths
        Output: List of AudioResult objects (one per input)
        All files share the same prompt and parameters.
        """

        logs_lines = []
        results = []

        try:
            # Validate input
            if not audio_files:
                raise ValueError("audio_files is empty")

            if not isinstance(audio_files, list):
                raise ValueError("audio_files must be a list")

            logs_lines.append(f"Processing {len(audio_files)} audio file(s)")

            # Process each audio
            for idx, url in enumerate(audio_files):
                logs_lines.append(f"[{idx+1}/{len(audio_files)}] {url[:80]}...")

                audio_path = None
                try:
                    # Download or use local
                    if url.startswith("http://") or url.startswith("https://"):
                        audio_path = self.download_audio_from_url(url)
                        logs_lines.append(f"  Downloaded to: {audio_path}")
                    else:
                        audio_path = url
                        logs_lines.append(f"  Using local: {audio_path}")

                    # Process
                    item_result = self.process_single_audio(
                        audio_path=audio_path,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        enable_thinking=enable_thinking,
                        temperature=temperature,
                        max_length=max_length,
                        embeddings_only=embeddings_only,
                        item_index=idx
                    )

                    results.append(item_result)

                    # Log result
                    if item_result.error:
                        logs_lines.append(f"  ✗ Error: {item_result.error[:100]}")
                    else:
                        if item_result.embeddings:
                            logs_lines.append(f"  ✓ Embeddings: {len(item_result.embeddings)} dims")
                        if item_result.embeddings_error:
                            logs_lines.append(f"  ✗ Emb error: {item_result.embeddings_error[:100]}")
                        if not embeddings_only and item_result.response:
                            logs_lines.append(f"  ✓ Response: {len(item_result.response)} chars")

                    # Cleanup temp file
                    if url.startswith("http") and audio_path and os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except:
                            pass

                except Exception as e:
                    error_msg = str(e)
                    logs_lines.append(f"  ✗ Failed: {error_msg}")
                    results.append(AudioResult(
                        index=idx,
                        embeddings=None,
                        embeddings_error=error_msg,
                        response="[error]",
                        error=error_msg
                    ))

                    if audio_path and os.path.exists(audio_path) and url.startswith("http"):
                        try:
                            os.remove(audio_path)
                        except:
                            pass

            logs_lines.append(f"Complete: {len(results)}/{len(audio_files)} processed")

            return ModelOutput(
                results=results,
                logs="\n".join(logs_lines)
            )

        except Exception as e:
            import traceback
            error_msg = str(e)
            trace = traceback.format_exc()
            logs_lines.append(f"ERROR: {error_msg}")
            logs_lines.append(trace)

            return ModelOutput(
                results=[AudioResult(
                    index=0,
                    embeddings=None,
                    embeddings_error=error_msg,
                    response="[error]",
                    error=f"{error_msg}\n{trace}"
                )],
                logs="\n".join(logs_lines)
            )
