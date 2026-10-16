# Live demo

`index.html` runs both detectors in the browser with onnxruntime-web and compares them
on your camera or an uploaded image.

Serve it over HTTP — opening the file directly blocks the model fetch:

```bash
python -m http.server 8777 --directory docs
# then open http://localhost:8777/
```

To publish it, enable GitHub Pages for this repository with `/docs` as the source.

## Regenerating the models

```bash
python utils/export_onnx.py --weights model_weights/blazeface.pth --out docs/blazeface_stock.onnx
python utils/export_onnx.py --weights runs/ckpt_peranchor/BlazeFace_final.pth --out docs/blazeface_tuned.onnx
```

The exported graph takes RGB `[1,3,128,128]` with pixels in `[0,255]` and returns decoded
boxes plus sigmoid scores; thresholding and NMS happen in JavaScript.

## One detail worth keeping

The page resamples the image to 128x128 by hand rather than letting the canvas do it.
Canvas downscaling box-averages the whole source footprint while `cv2.INTER_LINEAR`
samples a 2x2 neighbourhood, and at an 8x reduction those produce different images — enough
to move detections to different anchors entirely. With the manual resample the browser
reproduces the Python detector's boxes to within a pixel and its scores to within 0.02.
