"""Create a transparent reference image for Hunyuan3D shape inference."""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from hy3dshape.rembg import BackgroundRemover


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove an image background and save an RGBA PNG for shape inference."
    )
    parser.add_argument("input", type=Path, help="Source image containing one foreground object")
    parser.add_argument("output", type=Path, help="Transparent .png output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    if args.output.suffix.lower() != ".png":
        raise ValueError("Output must be a .png file so its alpha channel is preserved.")

    image = Image.open(args.input).convert("RGB")
    result = BackgroundRemover()(image).convert("RGBA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
