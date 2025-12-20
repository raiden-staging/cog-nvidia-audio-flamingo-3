# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import copy
import torch
import time
import subprocess
import json
from typing import Optional
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
    """
    Output schema - ALWAYS returns an array of results

    results_json: JSON array where each item has:
        - index: int
        - embeddings_json: str (JSON with {"vector": [...]} or {"error": "..."})
        - response: str (text response or "[embeddings_only mode]")
        - error: str or null
    """
    results_json: str
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
        Extract audio embeddings using the model's encoder.

        The dimensionality (1280) comes from the AF-Whisper encoder's hidden_size,
        NOT hardcoded - it's determined by the actual model architecture.

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
                # This returns features with shape [..., hidden_size]
                # where hidden_size is determined by the encoder architecture (1280 for AF-Whisper)
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

                # Check first element to detect nested structures
                if len(vec) > 0 and isinstance(vec[0], (list, tuple)):
                    return json.dumps({"error": "Nested structure detected"})

                # Success - return vector with its actual dimensionality
                return json.dumps({"vector": vec})

        except Exception as e:
            import traceback
            return json.dumps({"error": str(e), "trace": traceback.format_exc()})

    def download_audio_from_url(self, url: str) -> str:
        """Download audio from URL to temp file and return path"""
        import tempfile
        import urllib.request

        # Create temp file with appropriate extension
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

        # Download
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
    ) -> dict:
        """
        Process a single audio file and return result dict.

        Returns dict with:
        - index: int
        - embeddings_json: str (JSON with vector or error)
        - response: str (text response or "[embeddings_only mode]")
        - error: str or None
        """
        result = {
            "index": item_index,
            "embeddings_json": None,
            "response": None,
            "error": None
        }

        try:
            # STEP 1: Extract embeddings (always done)
            result["embeddings_json"] = self.extract_audio_embeddings(audio_path)

            # STEP 2: If embeddings_only, we're done
            if embeddings_only:
                result["response"] = "[embeddings_only mode]"
                return result

            # STEP 3: Generate text response
            sound = llava.Sound(audio_path)

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

            # Generate
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

            result["response"] = response
            return result

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            result["error"] = error_msg
            result["embeddings_json"] = json.dumps({"error": str(e)})
            result["response"] = "[error during processing]"
            return result

    def predict(
        self,
        audio_files: str = Input(
            description='JSON array of audio file URLs or paths. Example: ["https://example.com/audio1.mp3", "https://example.com/audio2.wav"]. Accepts http/https URLs or local paths.',
            default=""
        ),
        audio: Path = Input(
            description="DEPRECATED: Single audio file (kept for backward compatibility). Use audio_files instead.",
            default=None
        ),
        prompt: str = Input(
            description="Question or instruction about the audio. Applied to all audio files.",
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
        embeddings_only: bool = Input(
            description="If True, only return audio embeddings without generating text responses. Applies to all audio files.",
            default=False
        ),
    ) -> ModelOutput:
        """
        Analyze audio using Audio Flamingo 3.

        Input: JSON array of audio URLs/paths
        Output: JSON array of results (one per input audio)

        All audio files share the same prompt and parameters.
        """

        logs_lines = []
        results = []

        try:
            # Handle backward compatibility: if audio provided but not audio_files, convert to array
            if audio is not None and (not audio_files or not audio_files.strip()):
                audio_files = json.dumps([str(audio)])
                logs_lines.append("Using deprecated 'audio' parameter - converted to array format")

            # Parse audio_files JSON array
            audio_urls = json.loads(audio_files)

            # Validate it's a list
            if not isinstance(audio_urls, list):
                raise ValueError("audio_files must be a JSON array")

            if len(audio_urls) == 0:
                raise ValueError("audio_files array is empty")

            logs_lines.append(f"Processing {len(audio_urls)} audio file(s)")

            # Process each audio file
            for idx, url in enumerate(audio_urls):
                logs_lines.append(f"[{idx+1}/{len(audio_urls)}] Processing: {url[:80]}...")

                audio_path = None
                try:
                    # Download or use local path
                    if url.startswith("http://") or url.startswith("https://"):
                        audio_path = self.download_audio_from_url(url)
                        logs_lines.append(f"  Downloaded to: {audio_path}")
                    else:
                        audio_path = url
                        logs_lines.append(f"  Using local path: {audio_path}")

                    # Process this audio
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

                    # Add to results
                    results.append(item_result)

                    # Log result
                    if item_result["error"]:
                        logs_lines.append(f"  ✗ Error: {item_result['error'][:100]}")
                    else:
                        # Parse embeddings to show dims
                        try:
                            emb_data = json.loads(item_result["embeddings_json"])
                            if "vector" in emb_data:
                                logs_lines.append(f"  ✓ Embeddings: {len(emb_data['vector'])} dims")
                            elif "error" in emb_data:
                                logs_lines.append(f"  ✗ Embedding error: {emb_data['error'][:100]}")
                        except:
                            logs_lines.append(f"  ? Could not parse embeddings")

                        if not embeddings_only:
                            resp_len = len(item_result["response"]) if item_result["response"] else 0
                            logs_lines.append(f"  ✓ Response: {resp_len} chars")

                    # Clean up temp file if we downloaded it
                    if url.startswith("http") and audio_path and os.path.exists(audio_path):
                        try:
                            os.remove(audio_path)
                        except:
                            pass

                except Exception as e:
                    import traceback
                    error_msg = str(e)
                    logs_lines.append(f"  ✗ Failed: {error_msg}")

                    # Add error result
                    results.append({
                        "index": idx,
                        "embeddings_json": json.dumps({"error": error_msg}),
                        "response": "[error]",
                        "error": error_msg
                    })

                    # Clean up on error too
                    if audio_path and os.path.exists(audio_path) and url.startswith("http"):
                        try:
                            os.remove(audio_path)
                        except:
                            pass

            logs_lines.append(f"Completed: {len(results)}/{len(audio_urls)} items processed")

            # Return results as JSON array
            return ModelOutput(
                results_json=json.dumps(results, indent=2),
                logs="\n".join(logs_lines)
            )

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in audio_files: {str(e)}"
            logs_lines.append(f"ERROR: {error_msg}")

            return ModelOutput(
                results_json=json.dumps([{
                    "index": 0,
                    "embeddings_json": json.dumps({"error": error_msg}),
                    "response": "[json parse error]",
                    "error": error_msg
                }]),
                logs="\n".join(logs_lines)
            )

        except Exception as e:
            import traceback
            error_msg = str(e)
            trace = traceback.format_exc()
            logs_lines.append(f"ERROR: {error_msg}")
            logs_lines.append(trace)

            return ModelOutput(
                results_json=json.dumps([{
                    "index": 0,
                    "embeddings_json": json.dumps({"error": error_msg}),
                    "response": "[processing error]",
                    "error": f"{error_msg}\n{trace}"
                }]),
                logs="\n".join(logs_lines)
            )
