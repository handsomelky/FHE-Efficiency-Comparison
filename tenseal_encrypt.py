import argparse
import tenseal as ts
import numpy as np
from PIL import Image
import os
import sys
import time
import shutil



def save_context(context, file_path):
    with open(file_path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))

def load_context(file_path):
    print("Loading context from: ", file_path)
    with open(file_path, "rb") as f:
        return ts.Context.load(f.read())


def encrypt_image(image_path, algorithm, context, output_folder):
    # 加载图片并转换为 numpy 数组
    with Image.open(image_path) as img:
        img_data = np.array(img)

    # 将图片数据转换为一维数组
    img_data_flattened = img_data.flatten()


    start_time = time.time() 

    # 根据选择的算法加密图片数据
    if algorithm == 'BFV':
        encrypted_data = ts.bfv_vector(context, img_data_flattened.tolist())  # 加密整个向量
    elif algorithm == 'CKKS':
        encrypted_data = ts.ckks_vector(context, img_data_flattened.tolist())
    else:
        raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")

    end_time = time.time()
    encryption_time = end_time - start_time

    encrypted_data_serialize = encrypted_data.serialize()

    encrypted_filename = os.path.splitext(os.path.basename(image_path))[0] + "_encrypted"
    encrypted_file_path = os.path.join(output_folder, encrypted_filename)


    with open(encrypted_file_path, "wb") as f:
        f.write(encrypted_data_serialize)

    return encryption_time  # 返回加密所需时间


def encrypt_images_in_folder(input_folder, algorithm, context, output_folder):

    total_time = 0
    count = 0

     # 获取文件夹中符合条件的文件总数
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(files)
    last_percent_reported = None

    for i, filename in enumerate(files):
        image_path = os.path.join(input_folder, filename)
        encryption_time = encrypt_image(image_path, algorithm, context, output_folder)
        total_time += encryption_time
        count += 1

        # 更新进度条
        percent_done = (i + 1) / total_files
        bar_length = 50
        block = int(bar_length * percent_done )
        text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done*100)
        sys.stdout.write(text)
        sys.stdout.flush()

    # 当循环结束时，确保进度条显示为100%
    sys.stdout.write("\r[{}] 100.00%\n".format("█" * 50))
    sys.stdout.flush()

    return total_time, count

def main():
    parser = argparse.ArgumentParser(description="Encrypt images in a folder using TenSEAL.")
    parser.add_argument('-a', '--algorithm', type=str, choices=['BFV', 'CKKS'], required=True, help='Encryption algorithm (BFV or CKKS)')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Folder containing images to be encrypted')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Folder to save encrypted images')
    parser.add_argument('-p', '--poly_modulus_degree', type=int, default=32768, help='Polynomial modulus degree (default: 32768)')
    parser.add_argument('-m', '--plain_modulus', type=int, default=65537, help='Plain modulus for BFV algorithm (default: 65537)')
    parser.add_argument('-g', '--global_scale', type=float, default=2**40, help='Global scale for CKKS algorithm (optional, default: 2**40)')
    parser.add_argument('-c', '--context', type=str, help='Path to the serialized context file')
    args = parser.parse_args()

    keys_folder = "keys"  # 密钥存储目录
    if not os.path.exists(keys_folder):
        os.makedirs(keys_folder)

    # 创建加密上下文
    if args.context:
        context = load_context(args.context)

    else:
        if args.algorithm == 'BFV':
            context = ts.context(ts.SCHEME_TYPE.BFV, args.poly_modulus_degree, args.plain_modulus, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])
        elif args.algorithm == 'CKKS':
            context = ts.context(ts.SCHEME_TYPE.CKKS, args.poly_modulus_degree, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])

            context.global_scale = args.global_scale
   
        # 生成公私钥
        context.generate_galois_keys()

        # 序列化并保存包含私钥的上下文
        save_context(context, os.path.join(keys_folder, "private_context"))

        # 移除私钥
        context.make_context_public()

        # 序列化并保存公钥上下文
        save_context(context, os.path.join(keys_folder, "public_context"))

    # 创建保存加密数据的文件夹，如果已存在且包含文件，则先清空
    if os.path.exists(args.output_folder):
        # 清空文件夹
        for file in os.listdir(args.output_folder):
            file_path = os.path.join(args.output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting file {file_path}: {e}")
    else:
        # 创建文件夹
        os.makedirs(args.output_folder)

    # 展示加密时间
    print("=== Starting the encryption process ===")
    total_time, count = encrypt_images_in_folder(args.input_folder, args.algorithm, context, args.output_folder)
    
    print("=== Encryption process completed. ===\n")
    if count > 0:
        average_time = total_time / count
        print(f"Number of images encrypted: {count}")
        print(f"Total encryption time: {total_time:.3f} seconds")
        print(f"Average encryption time per image: {average_time:.3f} seconds")
    else:
        print("No images were encrypted.")

if __name__ == "__main__":
    main()
