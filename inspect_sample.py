import torch
from csv_dataloader import CSVDetectorDataset
from blazeface import BlazeFace

csv_path = 'data/splits/train_new.csv'
root_dir = 'data/raw/blazeface'
ds = CSVDetectorDataset(csv_path, root_dir, target_size=(128,128), augment=False)
sample = ds[0]
image = sample['image'].unsqueeze(0)
model = BlazeFace()
model.generate_anchors({'num_layers':4,'min_scale':0.1484375,'max_scale':0.75,'input_size_height':128,'input_size_width':128,'anchor_offset_x':0.5,'anchor_offset_y':0.5,'strides':[8,16,16,16],'aspect_ratios':[1.0],'reduce_boxes_in_lowest_layer':False,'interpolated_scale_aspect_ratio':1.0,'fixed_anchor_size':True})
from blazebase import load_mediapipe_weights
load_mediapipe_weights(model, 'model_weights/blazeface.pth')
model.eval()
with torch.no_grad():
    detections = model.predict_on_batch(image)[0]
print('detections', detections[:5])
print('gt first positive', sample['anchor_targets'][sample['anchor_targets'][:,0]==1][:1])

