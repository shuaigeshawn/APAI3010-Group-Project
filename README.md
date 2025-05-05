#  Introduction

This repository reproduces [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone). To align the results demonstrated by the original paper, we adopt various approaches and tricks, which may differ somewhat from the paper and another [implementation](https://github.com/guoqincode/Open-AnimateAnyone). 

It's worth noting that this is a very preliminary version, aiming for approximating the performance (roughly 80% under our test) showed in [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone). 

We will continue to develop it, and also welcome feedbacks and ideas from the community.


# 🎞️ Examples 

Here are some results we generated, with the resolution of 512x768.

https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/f0454f30-6726-4ad4-80a7-5b7a15619057

https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/337ff231-68a3-4760-a9f9-5113654acf48

<table class="center">
    
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/9c4d852e-0a99-4607-8d63-569a1f67a8d2" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/722c6535-2901-4e23-9de9-501b22306ebd" muted="false"></video>
    </td>
</tr>

<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/17b907cc-c97e-43cd-af18-b646393c8e8a" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/MooreThreads/Moore-AnimateAnyone/assets/138439222/86f2f6d2-df60-4333-b19b-4c5abcd5999d" muted="false"></video>
    </td>
</tr>
</table>

**Limitation**: We observe following shortcomings in current version:
1. The background may occur some artifacts, when the reference image has a clean background
2. Suboptimal results may arise when there is a scale mismatch between the reference image and keypoints. We have yet to implement preprocessing techniques as mentioned in the [paper](https://arxiv.org/pdf/2311.17117.pdf).
3. Some flickering and jittering may occur when the motion sequence is subtle or the scene is static.

These issues will be addressed and improved in the near future. We appreciate your anticipation!

# ⚒️ Installation

prerequisites: `3.11>=python>=3.8`, `CUDA>=11.3`, `ffmpeg` and `git`.

Python and Git:

- Python 3.10.11: https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
- git: https://git-scm.com/download/win

- Install [ffmpeg](https://ffmpeg.org/) for your operating system
  (https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
  
  notice:step 4 use windows system Set Enviroment Path.

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

```
git clone --recurse-submodules https://github.com/sdbds/Moore-AnimateAnyone-for-windows/
```

Install with Powershell run `install.ps1` or `install-cn.ps1`(for Chinese)

### Use local model

Add loading local safetensors or ckpt,you can change `config/prompts/animation.yaml` about `pretrained_weights` for your local SD1.5 model.
such as `"D:\\stablediffusion-webui\\models\\Stable-diffusion\\v1-5-pruned.ckpt"`

## No need Download models manually
~~Download weights~~

~~Download our trained [weights](https://huggingface.co/patrolli/AnimateAnyone/tree/main), which include four parts: `denoising_unet.pth`, `reference_unet.pth`, `pose_guider.pth` and `motion_module.pth`.~~

~~Download pretrained weight of based models and other components:~~ 
~~- [StableDiffusion V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)~~
~~- [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)~~
~~- [image_encoder](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main/image_encoder)~~

~~Download dwpose weights (`dw-ll_ucoco_384.onnx`, `yolox_l.onnx`) following [this](https://github.com/IDEA-Research/DWPose?tab=readme-ov-file#-dwpose-for-controlnet).~~

~~Put these weights under a directory, like `./pretrained_weights`, and orgnize them as follows:~~

```text
./pretrained_weights/
|-- DWPose
|   |-- dw-ll_ucoco_384.onnx
|   `-- yolox_l.onnx
|-- image_encoder
|   |-- config.json
|   `-- pytorch_model.bin
|-- denoising_unet.pth
|-- motion_module.pth
|-- pose_guider.pth
|-- reference_unet.pth
`-- stable-diffusion-v1-5
    |-- feature_extractor
    |   `-- preprocessor_config.json
    |-- model_index.json
    |-- unet
    |   |-- config.json
    |   `-- diffusion_pytorch_model.bin
    `-- v1-inference.yaml
```

~~Note: If you have installed some of the pretrained models, such as `StableDiffusion V1.5`, you can specify their paths in the config file (e.g. `./config/prompts/animation.yaml`).~~

# 🚀 Training and Inference 

## Inference

Here is the cli command for running inference scripts:

```shell
python -m scripts.pose2vid --config ./configs/prompts/animation.yaml -W 512 -H 784 -L 64
```

You can refer the format of `animation.yaml` to add your own reference images or pose videos. To convert the raw video into a pose video (keypoint sequence), you can run with the following command:

```shell
python tools/vid2pose.py --video_path /path/to/your/video.mp4
```

# 🎨 Gradio Demo 

### Local Gradio Demo:

Launch local gradio demo on GPU:

Powershell run with `run_gui.ps1`

Then open gradio demo in local browser.

### Online Gradio Demo:
## <span id="train"> Training </span>

Note: package dependencies have been updated, you may upgrade your environment via `pip install -r requirements.txt` before training.

### Data Preparation

Extract keypoints from raw videos: 

```shell
python tools/extract_dwpose_from_vid.py --video_root /path/to/your/video_dir
```

Extract the meta info of dataset:

```shell
python tools/extract_meta_info.py --root_path /path/to/your/video_dir --dataset_name anyone 
```

Update lines in the training config file: 

```yaml
data:
  meta_paths:
    - "./data/anyone_meta.json"
```

### Stage1

Put [openpose controlnet weights](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/tree/main) under `./pretrained_weights`, which is used to initialize the pose_guider.

Put [sd-image-variation](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/tree/main) under `./pretrained_weights`, which is used to initialize unet weights.

Run command:

```shell
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```

### Stage2

Put the pretrained motion module weights `mm_sd_v15_v2.ckpt` ([download link](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)) under `./pretrained_weights`. 

Specify the stage1 training weights in the config file `stage2.yaml`, for example:

```yaml
stage1_ckpt_dir: './exp_output/stage1'
stage1_ckpt_step: 30000 
```

Run command:

```shell
accelerate launch train_stage_2.py --config configs/train/stage2.yaml
```

**HuggingFace Demo**: We launch a quick preview demo of Moore-AnimateAnyone at [HuggingFace Spaces](https://huggingface.co/spaces/xunsong/Moore-AnimateAnyone)!!

We appreciate the assistance provided by the HuggingFace team in setting up this demo.

To reduce waiting time, we limit the size (width, height, and length) and inference steps when generating videos. 

If you have your own GPU resource (>= 16GB vram), you can run a local gradio app via following commands:

`python app.py`

# Our Extension
