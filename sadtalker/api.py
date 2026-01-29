"""
High-level SadTalker API for programmatic use.

Usage:
    from sadtalker.api import SadTalker

    st = SadTalker(checkpoint_path="path/to/checkpoints", device="cuda")
    frames = st.generate(pil_image, "audio.wav")
    # frames: list of PIL.Image
"""

import os
import tempfile

import numpy as np
from PIL import Image
from skimage import img_as_ubyte

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

from sadtalker.utils.init_path import init_path
from sadtalker.utils.preprocess import CropAndExtract
from sadtalker.test_audio2coeff import Audio2Coeff
from sadtalker.facerender.animate import AnimateFromCoeff
from sadtalker.generate_batch import get_data
from sadtalker.generate_facerender_batch import get_facerender_data
from sadtalker.facerender.modules.make_animation import make_animation

import torch


class SadTalker:
    """Wraps the 3-stage SadTalker pipeline into a single callable."""

    def __init__(self, checkpoint_path: str, device: str = "cuda", size: int = 256):
        config_dir = os.path.join(_PACKAGE_DIR, "config")
        sadtalker_paths = init_path(checkpoint_path, config_dir, size=size, preprocess="crop")

        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, device)
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
        self.device = device
        self.size = size

    @torch.inference_mode()
    def generate(self, image: Image.Image, audio_path: str, pose_style: int = 0) -> list[Image.Image]:
        """
        Generate talking-head video frames from a single face image and audio.

        Args:
            image: Source face image (PIL).
            audio_path: Path to audio file (.wav).
            pose_style: Head motion style index (0-45).

        Returns:
            List of PIL.Image frames (256×256 RGB).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save input image to disk (SadTalker expects file paths)
            pic_path = os.path.join(tmpdir, "source.png")
            image.save(pic_path)

            first_frame_dir = os.path.join(tmpdir, "first_frame")
            os.makedirs(first_frame_dir, exist_ok=True)

            # Stage 1: Crop and extract 3DMM coefficients
            first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
                pic_path, first_frame_dir, crop_or_resize="crop",
                source_image_flag=True, pic_size=self.size,
            )
            if first_coeff_path is None:
                raise RuntimeError("SadTalker: no face detected in source image")

            # Stage 2: Audio → motion coefficients
            batch = get_data(first_coeff_path, audio_path, self.device, ref_eyeblink_coeff_path=None)
            coeff_path = self.audio_to_coeff.generate(batch, tmpdir, pose_style)

            # Stage 3: Render frames (skip MP4 round-trip)
            data = get_facerender_data(
                coeff_path, crop_pic_path, first_coeff_path, audio_path,
                batch_size=2, size=self.size,
            )
            frames = self._render_frames(data)

        return frames

    def _render_frames(self, x: dict) -> list[Image.Image]:
        """Run the face renderer and return PIL frames directly (no MP4 I/O)."""
        afc = self.animate_from_coeff

        source_image = x["source_image"].float().to(self.device)
        source_semantics = x["source_semantics"].float().to(self.device)
        target_semantics = x["target_semantics_list"].float().to(self.device)

        yaw = x.get("yaw_c_seq")
        pitch = x.get("pitch_c_seq")
        roll = x.get("roll_c_seq")
        if yaw is not None:
            yaw = yaw.float().to(self.device)
        if pitch is not None:
            pitch = pitch.float().to(self.device)
        if roll is not None:
            roll = roll.float().to(self.device)

        predictions = make_animation(
            source_image, source_semantics, target_semantics,
            afc.generator, afc.kp_extractor, afc.he_estimator, afc.mapping,
            yaw, pitch, roll, use_exp=True,
        )
        predictions = predictions.reshape((-1,) + predictions.shape[2:])
        predictions = predictions[: x["frame_num"]]

        frames = []
        for idx in range(predictions.shape[0]):
            arr = predictions[idx].data.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            arr = img_as_ubyte(arr)
            frames.append(Image.fromarray(arr))

        return frames
