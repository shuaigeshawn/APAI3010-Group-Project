import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import argparse
import os

def load_model():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    return model

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

def resize_background(background, target_width, target_height):
    """Resize background to cover target dimensions while maintaining aspect ratio"""
    bg_height, bg_width = background.shape[:2]
    
    # Calculate scaling factors
    width_ratio = target_width / bg_width
    height_ratio = target_height / bg_height
    
    # Use the larger scaling factor to ensure full coverage
    scale = max(width_ratio, height_ratio)
    
    # Calculate new dimensions
    new_width = int(bg_width * scale)
    new_height = int(bg_height * scale)
    
    # Resize background
    resized_bg = cv2.resize(background, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculate crop positions (center)
    x = max(0, (new_width - target_width) // 2)
    y = max(0, (new_height - target_height) // 2)
    
    # Crop to target dimensions
    cropped_bg = resized_bg[y:y+target_height, x:x+target_width]
    
    return cropped_bg

def overlay_on_background(frame, mask, bg_frame):
    mask_3c = np.dstack([mask] * 3)
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    background = cv2.bitwise_and(bg_frame, bg_frame, mask=cv2.bitwise_not(mask))
    combined = cv2.add(foreground, background)
    return combined

def process_video(video_path, bg_video_path, output_path):
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model = load_model().to(device)
    
    # 打开输入视频和背景视频
    cap = cv2.VideoCapture(video_path)
    bg_cap = cv2.VideoCapture(bg_video_path)
    
    if not cap.isOpened() or not bg_cap.isOpened():
        print("Error opening video files")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 读取背景视频帧（自动循环）
        bg_ret, bg_frame = bg_cap.read()
        if not bg_ret:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bg_ret, bg_frame = bg_cap.read()
            if not bg_ret:
                break
        
        mask = segment_person(frame, model, device, 0.6, 1700)
        resized_bg = resize_background(bg_frame, width, height)
        combined_frame = overlay_on_background(frame, mask, resized_bg)
        out.write(combined_frame)
    
    cap.release()
    bg_cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change video background using segmentation.")
    parser.add_argument("video", type=str, help="Path to the input video.")
    parser.add_argument("bg_video", type=str, help="Path to the background video.")
    args = parser.parse_args()
    
    # Generate output path based on input filenames
    base_name = os.path.basename(args.video).split('.')[0]
    bg_name = os.path.basename(args.bg_video).split('.')[0]
    output_path = f"{base_name}_{bg_name}_output.mp4"
    
    process_video(args.video, args.bg_video, output_path)