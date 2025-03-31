import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision
import csv

def predict_cell_type(model_path, image_path, class_names):
    # 定义图像预处理方式
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 加载模型
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 读入并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0)

    # 前向传播并获取预测结果
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # 输出预测的细胞类型
    return class_names[predicted.item()]

if __name__ == '__main__':
    # 假设类名与训练时一致
    cell_classes = ['1', '2', '3']
    results = []

    for i in range(31):
        # 使用测试图像进行预测
        result = predict_cell_type(
            model_path="content/resnet50_scratch.pth",
            image_path=f"content/extract/{i}.png",
            class_names=cell_classes
        )
        results.append(result)
        print(f'Predicted cell type of {i}:', result)

    csv_path = "content/cell.csv"
    output_csv_path = "content/cell_predict.csv"

    with open(csv_path, 'r') as file:
        reader = list(csv.reader(file))

    for i, row in enumerate(reader):
        if i < len(results):
            row[0] = results[i]

    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(reader)