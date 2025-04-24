from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from django.conf import settings

# Load DETR model
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
detr_model.eval()

# Load Mask R-CNN model
maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn_model.eval()

def index(request):
    return render(request, 'index.html')

def detection(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(image_file.name, image_file)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Process image with DETR
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        inputs = detr_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = detr_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Draw bounding boxes
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [int(i) for i in box.tolist()]
            label_str = detr_model.config.id2label[label.item()]
            score_str = f"{score.item():.2f}"
            cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{label_str} {score_str}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save processed image
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
        cv2.imwrite(processed_path, image_cv)

        return render(request, 'detection_result.html', {
            'original': filename,
            'processed': processed_filename
        })

    return render(request, 'detection.html')

def segmentation(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(image_file.name, image_file)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Process image with Mask R-CNN
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            predictions = maskrcnn_model([image_tensor])[0]

        # Draw masks
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for i, (mask, score, label) in enumerate(zip(predictions['masks'], predictions['scores'], predictions['labels'])):
            if score > 0.5:  # Threshold for confidence
                mask = mask[0].mul(255).byte().numpy()
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                image_cv[mask > 128] = image_cv[mask > 128] * 0.5 + color * 0.5

        # Save processed image
        processed_filename = f"processed_{filename}"
        processed_path = os.path.join(settings.MEDIA_ROOT, processed_filename)
        cv2.imwrite(processed_path, image_cv)

        return render(request, 'segmentation_result.html', {
            'original': filename,
            'processed': processed_filename
        })

    return render(request, 'segmentation.html')