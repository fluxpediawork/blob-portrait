---
name: blob-portrait
description: Recreate any photo (a selfie, portrait, product shot) as tens of thousands of soft colour blobs trained from random noise by gradient descent, and output the painting, a photo-vs-code comparison, a zoom proof and a noise-to-picture GIF. Use when someone asks to rebuild, "recreate with code", or "gradient descent" an image, or wants a "which one is the photo?" reel.
---

# Blob portrait: gradient descent for a photo

Paints a photo with 2D Gaussian blobs. It starts from 1,500 random blobs, and each step it renders them, measures the error against the photo (L1 + SSIM), and moves every blob a little to reduce it (Adam). New tiny blobs spawn where the error is highest. No image generator is involved: the picture is learned.

## Steps

1. **Get the photo.** Use the attached file. Note its subject position so you can set `--focus` (the area that gets the most close-up training, usually the face).

2. **Install.** Python 3.10+.
   ```bash
   pip install -r requirements.txt   # torch, numpy, pillow
   ```
   If the PyTorch index is blocked, plain `pip install torch` from PyPI works.

3. **Pick a preset** from the hardware and time available:

   | preset | steps | max blobs | 2-core CPU | GPU |
   |---|---|---|---|---|
   | `quick` | 300 | 15,000 | ~8 min | ~1 min |
   | `default` | 1,600 | 45,000 | ~35 min | ~5 min |
   | `full` | 6,000 | 100,000 | hours | ~20-40 min |

   On a CPU, run `default` in the background and poll the log; don't block on it.

4. **Run.**
   ```bash
   python scripts/blobfit.py photo.jpg --out result/ --preset default \
       --aspect 4:5 --focus 0.5,0.4,0.45,0.5
   ```
   - `--aspect` crops to w:h around the centre (`none` keeps the photo's own shape).
   - `--focus cx,cy,w,h` is in 0-1 of the cropped image. For a centred selfie, `0.5,0.4,0.45,0.5` covers the face.
   - `--steps` / `--blobs` override the preset.
   - Device is picked automatically: CUDA, then Apple MPS, then CPU.

5. **Check the result before delivering.** Open `compare.jpg` and `zoom.jpg` and look at them. `stats.json` has the PSNR: around 28-30 dB is a soft likeness (quick), 38 dB+ is hard to tell apart at normal size. If the face is soft, rerun with more steps or move `--focus` onto the face.

6. **Deliver.** Send `compare.jpg`, `render.png` and `training.gif`. Say how many blobs and steps, the PSNR, and that it was trained, not generated.

## Outputs

| file | what it is |
|---|---|
| `render.png` | the blob painting at full fitted size |
| `compare.jpg` | real photo beside the code version |
| `zoom.jpg` | close-up proof around the eyes |
| `training.gif` | noise to picture, one frame per milestone |
| `blobs.npz` | every blob: position, scale, angle, colour, strength (for re-rendering or animation) |
| `stats.json` | steps, blob count, PSNR, seconds, device |

## Honest limits

- It copies one photo. It can't invent a new view or a new face.
- Quality depends on steps: short CPU runs come out soft; fine skin grain needs the `full` preset.
- The backdrop colour is taken from the photo's edges, so blobs only need to paint what differs from it.
