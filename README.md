AnimateAnyone-Reproduced
Consistent and Controllable Image-to-Video Synthesis for Character Animation
This repository contains our reproduction of the AnimateAnyone model, a framework for generating high-fidelity, temporally consistent character animations from a single reference image and pose sequences. Our implementation aims to closely replicate the original pipeline while introducing extensions for improved usability, flexibility, and performance. This project is intended for academic research and demonstration purposes.
🌟 Features

High-Fidelity Animation: Generates detailed, realistic character animations with consistent appearance across frames.
Pose-Guided Control: Uses pose sequences (e.g., OpenPose keypoints) to drive character movements.
Temporal Consistency: Ensures smooth transitions between frames using advanced temporal modeling.
Extended Features:
Support for multiple pose formats (OpenPose, DensePose, and custom JSON-based keypoints).
Integration with ComfyUI for a streamlined workflow.
Optional text prompt conditioning for stylized animations.
Batch processing for generating multiple videos simultaneously.


Pre-trained Weights: Includes unofficial pre-trained weights adapted from the original model.
Video Demonstrations: Showcases results for various character types (humans, cartoons, humanoid figures).

📋 Requirements

Python >= 3.10
CUDA >= 11.7 (for GPU acceleration)
GPU with at least 16GB VRAM (e.g., RTX 3080 or better)
Dependencies listed in requirements.txt

🛠 Installation

Clone the Repository:
git clone https://github.com/your-username/AnimateAnyone-Reproduced.git
cd AnimateAnyone-Reproduced


Set Up Virtual Environment (optional but recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

If you encounter issues with the diffusers library, run:
pip install --force-reinstall diffusers>=0.26.1


Download Pre-trained Weights:Run the following command to automatically download weights to the ./pretrained_weights directory:
python scripts/download_weights.py

Alternatively, manually download weights from our HuggingFace page and place them in ./pretrained_weights.


🚀 Usage
Basic Inference
To generate a character animation video, use the pose2vid script with a configuration file specifying the reference image, pose sequence, and output settings.
python -m scripts.pose2vid \
    --config ./configs/prompts/animation.yaml \
    --reference_image ./data/reference/chunli.png \
    --pose_sequence ./data/poses/chunli_poses/ \
    --output_dir ./outputs/ \
    --width 512 \
    --height 784 \
    --length 64

Arguments:

--config: Path to the configuration YAML file (see configs/prompts/animation.yaml for an example).
--reference_image: Path to the reference character image (PNG or JPG).
--pose_sequence: Directory containing pose sequence images (e.g., OpenPose keypoints).
--output_dir: Directory to save the generated video.
--width, --height: Output video resolution.
--length: Number of frames in the output video.

Extended Features
1. Multiple Pose Formats
Our implementation supports multiple pose formats:

OpenPose: Default format, as used in the original AnimateAnyone.
DensePose: For more detailed body surface mapping.
Custom JSON: Define keypoints in a JSON file for custom animations.

To use DensePose, preprocess your pose sequence with:
python scripts/preprocess_densepose.py --input_dir ./data/poses/ --output_dir ./data/densepose/

2. Text Prompt Conditioning
Add a text prompt to stylize the animation (e.g., "a cartoon character dancing in a futuristic city"):
python -m scripts.pose2vid \
    --config ./configs/prompts/animation.yaml \
    --reference_image ./data/reference/cartoon.png \
    --pose_sequence ./data/poses/cartoon_poses/ \
    --text_prompt "a cartoon character dancing in a futuristic city" \
    --output_dir ./outputs/

3. Batch Processing
Generate multiple videos in a single run by providing a directory of reference images and corresponding pose sequences:
python -m scripts.batch_pose2vid \
    --config ./configs/prompts/batch_animation.yaml \
    --input_dir ./data/batch_input/ \
    --output_dir ./outputs/batch/

4. ComfyUI Integration
For a visual workflow, integrate our model with ComfyUI:

Clone this repository into your ComfyUI custom_nodes directory:cd Your_ComfyUI_root_directory/ComfyUI/custom_nodes/
git clone https://github.com/your-username/AnimateAnyone-Reproduced.git


Install dependencies as above.
Launch ComfyUI and use the provided example workflow (workflows/animateanyone.json).

Example Configuration (animation.yaml)
model:
  denoising_unet: ./pretrained_weights/denoising_unet.pth
  motion_module: ./pretrained_weights/motion_module.pth
  pose_guider: ./pretrained_weights/pose_guider.pth
  reference_unet: ./pretrained_weights/reference_unet.pth
inference:
  steps: 20
  context_frames: 12
  guidance_scale: 7.5
output:
  fps: 30
  format: mp4

🎥 Video Demonstrations
Below are example videos generated using our model, showcasing its ability to animate diverse characters:

Full-Body Human AnimationInput: Reference image of a dancer, OpenPose sequenceOutput: Watch VideoDescription: A realistic human figure performing a dance sequence with smooth transitions and consistent details.

Cartoon Character AnimationInput: Cartoon character image, DensePose sequence, text prompt ("dancing in a neon city")Output: Watch VideoDescription: A stylized cartoon character animated with vibrant, context-aware backgrounds.

Humanoid Figure AnimationInput: Humanoid robot image, custom JSON keypointsOutput: Watch VideoDescription: A humanoid figure walking with precise pose control and minimal flickering.


Note: Replace placeholder links with actual video hosting URLs (e.g., YouTube, Vimeo, or your own server).
📊 Performance

Hardware: Tested on RTX 3080 (16GB VRAM).
Inference Time:
24 frames, 512x784, 20 steps, 12 context frames: ~425 seconds.
64 frames, 512x784, 20 steps, 24 context frames: ~835 seconds.


VRAM Usage:
256x256 resolution: ~11GB.
512x784 resolution: ~23.5GB.



🙏 Acknowledgments

The original AnimateAnyone team for their groundbreaking work.
Moore-AnimateAnyone for inspiration and implementation insights.
ComfyUI for the workflow integration.
HuggingFace for hosting pre-trained weights and demos.

📜 License
This project is licensed under the Apache License 2.0. See the LICENSE file for details. Note that this is an unofficial reproduction, and users are responsible for adhering to ethical and legal standards when using the model.
🤝 Contributing
We welcome contributions! Please check the CONTRIBUTING.md file for guidelines on submitting issues, pull requests, or new features.
📬 Contact
For questions or feedback, open an issue on this repository or contact us at [your-email@example.com].

This project is for academic research purposes only. We disclaim responsibility for user-generated content.

