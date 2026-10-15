"""Score stock and tuned against YuNet detections: a labeller neither model trained on."""
import os, sys, numpy as np, pandas as pd, cv2, torch
from pathlib import Path
REPO=Path("/mnt/14BE47C2BE479ADE/Code/ear_stuff/trainable_blazeface"); sys.path.insert(0,str(REPO))
from utils.compare_detectors import iou_matrix
from utils.debug_training import _run_detector_on_image
from utils.model_utils import load_model

# Independent labeller. Download once:
#   curl -L -o yunet.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
MODEL = os.environ.get("YUNET_ONNX", "yunet.onnx")
N=2500; CONF=0.80; THR=0.5
root=REPO/"data/raw/blazeface"
df=pd.read_csv(REPO/"data/splits/val.csv")
groups=list(df.groupby("image_path",sort=False))
rng=np.random.default_rng(0)
groups=[groups[i] for i in sorted(rng.choice(len(groups),size=N,replace=False).tolist())]
dev=torch.device("cuda")

det=cv2.FaceDetectorYN.create(MODEL,"",(320,320),CONF,0.3,5000)
# Independent labels first, so both detectors are judged against the identical set.
labels={}
for rel,_ in groups:
    im=cv2.imread(str(root/str(rel)))
    if im is None: continue
    h,w=im.shape[:2]
    det.setInputSize((w,h))
    _,faces=det.detect(im)
    if faces is None or len(faces)==0:
        labels[rel]=np.zeros((0,4),np.float32); continue
    f=np.asarray(faces,np.float32)
    labels[rel]=np.column_stack((f[:,0],f[:,1],f[:,0]+f[:,2],f[:,1]+f[:,3])).astype(np.float32)
tot=sum(len(v) for v in labels.values())
print(f"YuNet labelled {tot} faces over {len(labels)} images (conf {CONF})\n")

WEIGHTS = sys.argv[1:] or ["stock=model_weights/blazeface.pth"]
for spec in WEIGHTS:
    lbl, path = spec.split("=", 1)
    m=load_model(str(REPO/path),dev,threshold=THR)
    for iou_t in (0.3,0.5):
        tp=fp=0; ngt=0
        for rel,_ in groups:
            if rel not in labels: continue
            im=cv2.imread(str(root/str(rel)))
            if im is None: continue
            gt=labels[rel]; ngt+=len(gt)
            b,s=_run_detector_on_image(m,cv2.cvtColor(im,cv2.COLOR_BGR2RGB),dev)
            b=b[s>=THR]
            M=iou_matrix(b,gt)
            found=(M.max(axis=0)>=iou_t) if M.size else np.zeros(len(gt),bool)
            tp+=int(found.sum())
            fp+=len(b)-(int((M.max(axis=1)>=iou_t).sum()) if M.size else 0)
        print(f"{lbl:26} IoU{iou_t}: precision {tp/max(tp+fp,1):.4f}  recall {tp/max(ngt,1):.4f}")
