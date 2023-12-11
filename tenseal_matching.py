import argparse
import tenseal as ts
import sqlite3
import numpy as np
from PIL import Image
import os
import time
import sys
import json

def load_context(file_path):
    with open(file_path, "rb") as f:
        return ts.Context.load(f.read())

def load_encrypted_data_from_db(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, encrypted_data FROM images")
    encrypted_items = cursor.fetchall()
    conn.close()
    return encrypted_items

def distance_to_similarity(distance, max_distance):
    if distance >= max_distance:
        return 0
    else:
        return (1 - distance / max_distance) * 100

def compute_euclidean_distance(enc1, enc2, context):
    sk = context.secret_key()
    
    # 计算两个向量的差异
    diff_enc = enc1 - enc2
    diff = diff_enc.decrypt(sk)
    # 计算差异的平方
    diff_squared = [x**2 for x in diff]
    diff_squared_sum = sum(diff_squared)

    return diff_squared_sum


def decrypt_image(encrypted_data, algorithm, context):
    start_time = time.time()
    # 反序列化并解密数据
    if algorithm == 'BFV':
        decrypted_data = encrypted_data.decrypt()
    elif algorithm == 'CKKS':
        decrypted_data = encrypted_data.decrypt()
    else:
        raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")

    end_time = time.time()  # 结束计时
    decryption_time = end_time - start_time  # 计算解密时间

    # 转换解密数据为图片格式
    decrypted_image = np.array(decrypted_data, dtype=np.uint8).reshape((64, 64, -1))
    return decrypted_image, decryption_time

def get_image_tags(db_name, image_filename):
    """
    从数据库中检索图像的标签信息。
    假设数据库中有一个表格存储标签，其中包含身份标签和特征标签列。
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("SELECT identity, attributes FROM images WHERE filename = ?", (image_filename,))
    result = cursor.fetchone()
    conn.close()

    if result:
        identity_tag, feature_tags_json = result
        # 将 JSON 格式的特征标签转换为字典
        feature_tags_dict = json.loads(feature_tags_json)
        # 仅选择值为 '1' 的标签
        selected_features = [tag for tag, value in feature_tags_dict.items() if value == '1']
        return identity_tag, selected_features
    return None, []

def match_and_decrypt_images(encrypted_folder, db_name, algorithm, context, output_folder):
    encrypted_items = load_encrypted_data_from_db(db_name)
    total_db_images = len(encrypted_items)
    max_distance = 255*255*64*64

    for query_filename in os.listdir(encrypted_folder):
        encrypted_query_path = os.path.join(encrypted_folder, query_filename)
        with open(encrypted_query_path, "rb") as f:
            encrypted_query = f.read()

        print(f"=== Matching image: {query_filename} ===")
        min_distance = max_distance
        best_match = None

        if algorithm == 'BFV':
            encrypted_query = ts.bfv_vector_from(context, encrypted_query)
        elif algorithm == 'CKKS':
            encrypted_query = ts.ckks_vector_from(context, encrypted_query)
        else:
            raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")

        start_time = time.time()
        for i, (db_filename, encrypted_db_data) in enumerate(encrypted_items):
            if algorithm == 'BFV':
                encrypted_db_data = ts.bfv_vector_from(context, encrypted_db_data)
            elif algorithm == 'CKKS':
                encrypted_db_data = ts.ckks_vector_from(context, encrypted_db_data)
            else:
                raise ValueError("Unsupported algorithm. Please choose 'BFV' or 'CKKS'.")
            distance = compute_euclidean_distance(encrypted_query, encrypted_db_data, context)
            if distance < min_distance:
                min_distance = distance
                best_match = (db_filename, encrypted_db_data)
            # 更新进度条
            percent_done = (i + 1) / total_db_images
            bar_length = 50
            block = int(bar_length * percent_done)
            sys.stdout.write(f"\r[{'█' * block}{' ' * (bar_length - block)}] {percent_done * 100:.2f}%")
            sys.stdout.flush()
        sys.stdout.write('\n')
        end_time = time.time()
        total_time = end_time - start_time
        average_time_per_image = total_time / total_db_images if total_db_images > 0 else 0
        print(f"Matching completed in {total_time:.2f} seconds.")
        print(f"Average time per image: {average_time_per_image:.3f} seconds.")

        if best_match:
            decrypted_image,decryption_time = decrypt_image(best_match[1], algorithm, context)
            similarity = distance_to_similarity(min_distance,max_distance)
            output_filename = os.path.splitext(query_filename)[0] + "_matched_with_" + os.path.splitext(best_match[0])[0] + ".jpg"
            output_file_path = os.path.join(output_folder, output_filename)
            Image.fromarray(decrypted_image).save(output_file_path)
            print(f"Decryption time for the matched image: {decryption_time:.3f} seconds")
            print(f"Query image '{query_filename}' best matches with database image '{best_match[0]}'")
            print(f"Similarity is '{similarity:.2f}%'")
            identity_tag, feature_tags = get_image_tags(db_name, best_match[0])
            print(f"Identity Tag: {identity_tag}")
            print(f"Feature Tags: {', '.join(feature_tags)}\n")
            print("=== Matching and decryption completed. ===\n")

def main():
    parser = argparse.ArgumentParser(description="Match and decrypt images using TenSEAL.")
    parser.add_argument('-a', '--algorithm', type=str, choices=['BFV', 'CKKS'], required=True, help='Decryption algorithm (BFV or CKKS)')
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Folder containing encrypted data')
    parser.add_argument('-db', '--database', type=str, required=True, help='SQLite database file name')
    parser.add_argument('-o', '--output_folder', type=str, required=True, help='Folder to save decrypted images')
    parser.add_argument('-c', '--context', type=str, required=True, help='Path to the serialized context file containing the private key')

    args = parser.parse_args()

    context = load_context(args.context)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    match_and_decrypt_images(args.input_folder, args.database, args.algorithm, context, args.output_folder)

if __name__ == "__main__":
    main()
