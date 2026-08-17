# Assets

Source art and screen recordings. Referenced from [the README](https://github.com/martinkorelic/mobiletransformers/blob/main/README.md) and the docs pages.

| file | used by |
| --- | --- |
| `mobiletransformers_banner.png` | the README header, every model card, the Hugging Face org card |
| `mobiletransformers_banner_small.png` | **nothing in this repository.** Kept because a Hugging Face org/model card may reference it by URL, which a grep here cannot see. Delete it if not. |
| `mobiletransformers_logo.png` | **the source the Android icons were cut from** — see below |
| `mobiletransformers_train.gif` | README ▸ Examples · `SHOWCASE.md` ▸ Training — a LoRA run and the merge that follows it |
| `mobiletransformers_functioncall.gif` | README ▸ Examples · `SHOWCASE.md` ▸ Chat — a tool call validated and fired as a real alarm |
| `mobiletransformers_rag.gif` | README ▸ Examples · `SHOWCASE.md` ▸ Retrieval — grounded answering, sources shown first |
| `mobiletransformers_classify.gif` | README ▸ Examples · `SHOWCASE.md` ▸ Classify — a sentiment encoder scoring text on device |

## Recording a showcase clip

Still unrecorded, and marked as `<!-- CLIP: … -->` placeholders in `SHOWCASE.md`: the drawer changing
shape per package, and install-from-catalog. Each placeholder says what to capture.

The four committed clips are **800px wide and 1–7 MB**, above the guidance below. They were kept at
capture resolution deliberately: the phone UI's body text stops being legible when scaled to 480, and
an unreadable screenshot of a text-heavy screen demonstrates nothing. Treat the numbers below as the
target for a *new* clip, not as a rule to retrofit onto these.

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
