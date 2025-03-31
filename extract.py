import cv2
import numpy as np
import os
import csv

def add_first_row(csv_file, new_row, output_file):
    with open(csv_file, 'r') as file:
        reader = list(csv.reader(file))  # Read all rows into a list

    # Add the new row at the top
    updated_rows = [new_row] + reader

    # Write the updated content to a new CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

def extract_and_save_cells(mask_path, image_path, output_dir, csv_path):


    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    os.makedirs(output_dir, exist_ok=True)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # 如果CSV文件不存在，则写入表头
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["file_name", "x", "y", "w", "h"])

        n = 0
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 50:
                continue
            cell_image = image[y:y + h, x:x + w]
            output_path = os.path.join(output_dir, f"{n}.png")
            cv2.imwrite(output_path, cell_image)
            # 记录文件名与位置
            writer.writerow([f"{n}.png", x, y, w, h])
            n += 1
    return n

mask_path = "content/seg/mask.png"  # Path to segmentation mask
image_path = "content/Cyto_Urine_24_180325_01.jpeg"  # Path to corresponding image
output_dir = "content/extract"  # Directory to save cropped cell images

extract_and_save_cells(mask_path, image_path, output_dir,"content/cell.csv")