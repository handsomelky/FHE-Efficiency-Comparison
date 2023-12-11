from PIL import Image
import os

def resize_images(folder_path, output_folder, output_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 或者根据实际情况调整文件格式
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, filename)

            with Image.open(image_path) as img:
                min_side = min(img.width, img.height)
                # 计算裁剪区域
                left = (img.width - min_side) / 2
                top = (img.height - min_side) / 2
                right = left + min_side
                bottom = top + min_side

                # 裁剪和缩放
                img_cropped = img.crop((left, top, right, bottom))
                img_resized = img_cropped.resize((output_size))

                # 保存处理后的图片
                img_resized.save(output_path)

resize_images('data/matching_imgs', 'data/matching_imgs')