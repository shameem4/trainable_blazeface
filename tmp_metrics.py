import torch
from csv_dataloader import get_csv_dataloader
from blazebase import generate_reference_anchors
from loss_functions import BlazeFaceDetectionLoss
from train_blazeface import BlazeFaceTrainer, create_model

csv_path = 'data/splits/train_new.csv'
root_dir = 'data/raw/blazeface'
loader = get_csv_dataloader(csv_path, root_dir, batch_size=4, shuffle=False, num_workers=0, augment=False)
batch = next(iter(loader))

model = create_model(init_weights='mediapipe', weights_path='model_weights/blazeface.pth')
model.eval()
images = batch['image']
with torch.no_grad():
    raw_boxes, raw_scores = model.get_training_outputs(images)
class_predictions = torch.sigmoid(raw_scores)
anchor_predictions = raw_boxes[..., :4]

loss_fn = BlazeFaceDetectionLoss()
trainer_dummy = BlazeFaceTrainer(model=model, train_loader=loader, loss_fn=loss_fn, device='cpu')
metrics = trainer_dummy._compute_metrics(class_predictions, batch['anchor_targets'], anchor_predictions)
print(metrics)

