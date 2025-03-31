import cv2
import csv

def mark_cells(csv_path, image_path, output_path):
    image = cv2.imread(image_path)
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            cell_name = row[0]
            x = int(row[1])
            y = int(row[2])
            w = int(row[3])
            h = int(row[4])
            cell_type=row[0]
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, cell_type, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(output_path, image)


def resize_image(input_path, output_path, size=(256, 256)):
    image = cv2.imread(input_path)
    resized_image = cv2.resize(image, size)
    cv2.imwrite(output_path, resized_image)

resize_image("content/overlay_image.jpg", "content/resized_image.jpg")

mark_cells("content/cell_predict.csv", "content/resized_image.jpg", "content/marked_cells.jpg")