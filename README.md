# blob-portrait

**Recreate any photo with code: tens of thousands of colour blobs that learn, by gradient descent, to become the picture.**

One photo goes in. The code starts from random noise, and every step it paints the blobs, measures how wrong the picture is, and nudges each blob to be a little less wrong. No diffusion model, no image generator.

This repo is both a standalone script and a Claude skill (`SKILL.md`).

## Run it

```bash
pip install -r requirements.txt
python scripts/blobfit.py my-photo.jpg --out result/                  # ~35 min on a 2-core CPU
python scripts/blobfit.py my-photo.jpg --out result/ --preset quick   # ~8 min smoke test
python scripts/blobfit.py my-photo.jpg --out result/ --preset full    # best quality, use a GPU
```

You get `render.png`, `compare.jpg` (photo vs code), `zoom.jpg`, `training.gif` (noise → picture), `blobs.npz` and `stats.json`.

Useful flags: `--aspect 4:5` (crop shape, or `none`), `--focus cx,cy,w,h` (the area, in 0–1, that gets the most close-up training, usually the face), `--steps`, `--blobs`, `--device`.

## How it works

1. **A blob** is a soft see-through Gaussian: position (2), width and height (2), angle (1), colour (3), strength (1).
2. **Start random:** 1,500 blobs with random places, sizes and colours.
3. **Paint** them into an image (a hand-written differentiable splat renderer in PyTorch).
4. **Measure** the mistake against the photo: L1 plus SSIM.
5. **Nudge** every blob with Adam. Only blobs in view move.
6. **Coarse to fine:** whole image at 160 px, then 320 px, then random close-up crops at mid and full size, biased toward the face.
7. **More blobs where it's wrong:** every few steps, tiny new blobs spawn at the pixels with the biggest error.

## Use as a Claude skill

Put this folder where your Claude setup loads skills (or upload it as a skill), then ask: *"recreate this photo with the blob-portrait skill"*.

## Credit

Inspired by Sirio Berati's [gradient-descent-for-a-face](https://github.com/sirioberati/gradient-descent-for-a-face). This is an independent implementation of the same idea, generalised to any photo, with no code copied from that repo.

## License

MIT
