import sqlite3
import argparse
import os
import sys
import json
import re

def create_database(db_name):

    print(f"=== Creating database:{db_name} ===", )
    if os.path.exists(db_name):
        # 如果存在，则删除该文件
        os.remove(db_name)
        print("Existing database removed.")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            filename TEXT PRIMARY KEY,
            identity TEXT,
            attributes TEXT,
            encrypted_data BLOB
        )
    ''')
    conn.commit()
    conn.close()
    print("=== Database created successfully. ===\n")

def parse_identity_file(file_path):
    identity = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            identity[parts[0]] = parts[1]
    return identity

def parse_attr_file(file_path):
    attributes = {}
    with open(file_path, "r") as file:
        # 读取并忽略第一行
        next(file)
        attr_names = file.readline().strip().split()
        for line in file:
            parts = line.strip().split()
            attributes[parts[0]] = dict(zip(attr_names, parts[1:]))
    return attributes

def insert_into_database(db_name, filename, identity, attributes, encrypted_data):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    attr_values = json.dumps(attributes)

    cursor.execute('''
        INSERT INTO images (filename, identity, attributes, encrypted_data)
        VALUES (?, ?, ?, ?)
    ''', [filename, identity, attr_values, encrypted_data])

    conn.commit()
    conn.close()



def process_images(input_folder, db_name, identity_data, attr_data):
    print(f"=== Processing images in folder:{input_folder} ===" )
    files = os.listdir(input_folder)
    total_files = len(files)
    for i, filename in enumerate(files):
        file_number = re.sub('[^0-9]', '', filename)
        image_filename = file_number + '.jpg'
        identity = identity_data.get(image_filename)
        attributes = attr_data.get(image_filename)
        if identity is not None and attributes is not None:
            image_path = os.path.join(input_folder, filename)
            with open(image_path, "rb") as f:
                encrypted_data = f.read()
            insert_into_database(db_name, image_filename, identity, attributes, encrypted_data)
        
        # 更新进度条
        percent_done = (i + 1) * 100 / total_files
        bar_length = 50
        block = int(bar_length * percent_done / 100)
        text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
        sys.stdout.write(text)
        sys.stdout.flush()
    
    sys.stdout.write("\r[{}] 100.00%\n".format("█" * 50))
    sys.stdout.flush()
    print("=== Data inserted into the database. ===")


def main():
    parser = argparse.ArgumentParser(description="Process encrypted images and store in a SQLite database.")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help='Folder containing encrypted data')
    parser.add_argument('-db', '--database', type=str, default='encrypted_images.db', help='SQLite database file name')
    args = parser.parse_args()

    # 创建数据库
    create_database(args.database)

    # 解析标识和属性文件
    identity_data = parse_identity_file("Anno/identity_CelebA.txt")
    attr_data = parse_attr_file("Anno/list_attr_celeba.txt")

    # 处理图像
    process_images(args.input_folder, args.database, identity_data, attr_data)

if __name__ == "__main__":
    main()