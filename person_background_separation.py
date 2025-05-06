import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from PIL import Image
import argparse
import os

# 加载 DeepLabV3 模型
def load_model():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    return model

# 执行语义分割
def segment_person(frame, model, device, threshold, min_area):
    original_height, original_width = frame.shape[:2]
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    # 获取人像概率图
    output_softmax = torch.softmax(output, dim=0)
    person_prob = output_softmax[15].cpu().numpy()  # 15为PASCAL VOC的人物类别

    # 调整概率图到原始尺寸
    resized_prob = cv2.resize(person_prob, (original_width, original_height), 
                               interpolation=cv2.INTER_LINEAR)

    # 应用阈值处理
    person_mask = ((resized_prob > threshold) * 255).astype(np.uint8)

    # 形态学开运算（去除小噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)

    # 连通区域过滤
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(person_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask

def extract_and_save_person_and_background(frame, mask, output_person_path, output_background_path):
    # 将OpenCV的BGR格式转为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 创建包含Alpha通道的RGBA图像
    rgba_person = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    rgba_person[..., :3] = rgb_frame  # RGB通道
    rgba_person[..., 3] = (mask == 255).astype(np.uint8) * 255  # Alpha通道

    # 仅背景
    background_mask = (mask == 0).astype(np.uint8) * 255
    rgba_background = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    rgba_background[..., :3] = rgb_frame  # RGB通道
    rgba_background[..., 3] = background_mask  # Alpha通道

    # 保存为PNG格式
    output_person_image = Image.fromarray(rgba_person)
    output_person_image.save(output_person_path)
    
    output_background_image = Image.fromarray(rgba_background)
    output_background_image.save(output_background_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract person and background from an image.")
    parser.add_argument("input_image", type=str, help="Path to the input image.")
    args = parser.parse_args()

    device = torch.device('mps' if torch.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model().to(device)

    # 读取输入图像
    input_image = cv2.imread(args.input_image)
    if input_image is None:
        raise ValueError("无法打开输入图像文件。")

    # 执行分割
    mask = segment_person(input_image, model, device, 0.5, 1700)

    # 生成输出文件路径
    base_name = os.path.splitext(args.input_image)[0]
    output_person_path = f"{base_name}_person.png"
    output_background_path = f"{base_name}_background.png"

    # 输出文件路径
    extract_and_save_person_and_background(input_image, mask, output_person_path, output_background_path)

    print(f"提取完成，人物图像保存为: {output_person_path}，背景图像保存为: {output_background_path}")