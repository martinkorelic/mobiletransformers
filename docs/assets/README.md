# Assets

Source art and screen recordings. Referenced from [the README](../../README.md) and the docs pages.

| file | used by |
| --- | --- |
| `mobiletransformers_banner.png` | the README header, every model card, the Hugging Face org card |
| `mobiletransformers_logo.png` | **the source the Android icons were cut from** — see below |
| `base-model.gif` / `on-device-trained.gif` | README ▸ See it work. **Stale** — predate the app rewrite; to be replaced by a single `on-device-finetune.gif` |
| `ortransformer-feature.gif` | **nothing — orphaned.** Predates the app rewrite; delete when the new clips land |

## Recording a showcase clip

Clips still to make: `on-device-finetune.gif` and `offline-generation.gif` for the README, plus the
per-capability ones marked in `SHOWCASE.md`. Each placeholder there says what to capture.

**One claim per clip, and the claim must be visible without the caption.** A reader sees motion,
decides in two seconds, and scrolls. Aim for 6–12 seconds and under 5 MB — longer and GitHub
lazy-loads it into a grey box; larger and it never finishes loading on a phone.

```bash
adb shell screenrecord --size 720x1520 --bit-rate 8M --time-limit 30 /sdcard/shot.mp4
adb pull /sdcard/shot.mp4

# mp4 -> gif. The generated palette matters: a default-palette gif of this dark UI bands badly.
ffmpeg -i shot.mp4 -vf "fps=12,scale=480:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i shot.mp4 -i palette.png \
  -lavfi "fps=12,scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif
```

`screenrecord` caps at 3 minutes and **does not capture the touch indicator** — turn on
*Developer options ▸ Show taps* first, or things happen with no visible cause. Trim dead air: the
first frame is the thumbnail, so make it the state *before* the interesting thing. And check the
status bar before recording — that frame goes on the internet.

## Replacing the logo

The Android icon resources were cut from `mobiletransformers_logo.png` and are **committed as
finished artwork** — there is no generator, so swapping the logo means redoing them. These are the
files, and the four rules that were applied to produce them.

Under `android/MobileTransformers/MobileTransformersApp/src/main/res/`:

| file | densities | what it is |
| --- | --- | --- |
| `mipmap-*/ic_launcher_foreground.png` | m/h/xh/xxh/xxxh | adaptive-icon foreground, 108dp canvas |
| `mipmap-*/ic_launcher_monochrome.png` | m/h/xh/xxh/xxxh | Android 13+ themed-icon layer |
| `mipmap-*/ic_launcher.webp` + `ic_launcher_round.webp` | m/h/xh/xxh/xxxh | legacy pre-API-26 icons, 48dp |
| `drawable-*/ic_logo.png` | m/h/xh/xxh/xxxh | the top-app-bar mark, 32dp |
| `values/ic_launcher_background.xml` | — | the adaptive-icon background colour |

**1. Cut the alpha noise floor first.** This source carries roughly 51,000 pixels at alpha 1–31 — an
artefact of how it was produced. They are invisible against white and become a dirty haze the instant
the art is composited onto the dark launcher background. Zero every pixel at **alpha ≤ 32**, and zero
its RGB too: a transparent pixel still carries colour, and resampling blends it back in as a dark
halo. Real antialiased edges live at alpha 32–255 and must be left alone.

**2. Scale the foreground into the 66dp safe zone.** Android composites an adaptive icon on a 108dp
canvas and lets the launcher mask it to a circle, squircle or rounded square — only the centre
**66dp** is guaranteed to survive. Art that fills its canvas loses its edges on most phones. The
committed foreground occupies **60%** of the canvas width, centred, which clears a circular mask with
margin to spare.

**3. The background is dark on purpose.** `#171E22`, taken from the logo's own outline. The mark has a
white sticker outline that disappears completely on a light background — on white the icon reads as a
yellow blob with no silhouette.

**4. The monochrome layer is line art, not a silhouette.** A filled silhouette of a sticker is a
featureless blob. The committed layer keeps only the *dark stroke* pixels (alpha > 128 and luminance
< 110), which preserves the outlines, eyes, smile, phone frame and brain traces — recognisable at
icon size, where a blob is not.

Resample with Lanczos. Do not reach for ImageMagick's SVG path on this machine: its delegate points at
`rsvg-convert`, which is not installed, so it silently falls back to a weaker renderer.
