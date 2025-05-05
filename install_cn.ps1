Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1

if (!(Test-Path -Path "venv")) {
    Write-Output  "����python���⻷��venv..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "��װ����..."
pip install -U -r requirements-windows.txt -i https://mirror.baidu.com/pypi/simple

Write-Output "���ģ��..."

if (!(Test-Path -Path "pretrained_weights")) {
    Write-Output  "����ģ���ļ��в�����ģ��..."
    git lfs install
    git lfs clone https://huggingface.co/patrolli/AnimateAnyone pretrained_weights
    if (Test-Path -Path "pretrained_weights/.git/lfs") {
        Remove-Item -Path pretrained_weights/.git/* -Recurse -Force
    }
}

Set-Location .\pretrained_weights

if (!(Test-Path -Path "image_encoder")) {
    Write-Output  "����image_encoderģ��..."
    git lfs install
    git lfs clone https://huggingface.co/bdsqlsz/image_encoder
    if (Test-Path -Path "image_encoder/.git/lfs") {
        Remove-Item -Path image_encoder/.git/* -Recurse -Force
    }
}

$install_SD15 = Read-Host "�Ƿ���Ҫ����huggingface��SD15ģ��? ��������û���κ�SD15ģ��ѡ��y�������Ҫ������SD1.5ģ��ѡ�� n��[y/n] (Ĭ��Ϊ y)"
if ($install_SD15 -eq "y" -or $install_SD15 -eq "Y" -or $install_SD15 -eq "") {
    if (!(Test-Path -Path "stable-diffusion-v1-5")) {
        Write-Output  "���� stable-diffusion-v1-5 ģ��..."
        git lfs clone https://huggingface.co/bdsqlsz/stable-diffusion-v1-5
        
    }
    if (Test-Path -Path "stable-diffusion-v1-5/.git/lfs") {
        Remove-Item -Path stable-diffusion-v1-5/.git/lfs/* -Recurse -Force
    }
}


if (!(Test-Path -Path "DWPose")) {
    Write-Output  "���� dwpose ģ��..."
    mkdir "DWPose"
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx -o DWPose/dw-ll_ucoco_384.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx -o DWPose/yolox_l.onnx
}

Write-Output "��װVideo_controlnet_aux..."

git submodule update --recursive --init

Set-Location $PSScriptRoot/video_controlnet_aux
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install -r requirements-video.txt -i https://mirror.baidu.com/pypi/simple

Write-Output "��װ���"
Read-Host | Out-Null ;
