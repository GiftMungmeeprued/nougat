"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import sys
from pathlib import Path
import logging
import re
import argparse
import re
from functools import partial
import torch
from tqdm import tqdm
from nougat import NougatModel
from nougat.utils.dataset import LazyImageDataset
from nougat.utils.device import move_to_device, default_batch_size
from nougat.utils.checkpoint import get_checkpoint
from nougat.postprocessing import markdown_compatible
import time

logging.basicConfig(level=logging.WARN)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=default_batch_size(),
        help="Batch size to use.",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to checkpoint directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="0.1.0-base",
        help=f"Model tag to use.",
    )
    parser.add_argument("--out", "-o", type=Path, help="Output directory.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute already computed PDF, discarding previous predictions.",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of bfloat16. Can speed up CPU conversion for some setups.",
    )
    parser.add_argument(
        "--no-markdown",
        dest="markdown",
        action="store_false",
        help="Do not add postprocessing step for markdown compatibility.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Add postprocessing step for markdown compatibility (default).",
    )
    parser.add_argument(
        "--no-skipping",
        dest="skipping",
        action="store_false",
        help="Don't apply failure detection heuristic.",
    )
    parser.add_argument("input", type=Path, help="Path to PNGs to process.")
    args = parser.parse_args()
    if args.checkpoint is None or not args.checkpoint.exists():
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
    if args.out is None:
        logging.warning("No output directory. Output will be printed to console.")
    else:
        if not args.out.exists():
            logging.info("Output directory does not exist. Creating output directory.")
            args.out.mkdir(parents=True)
        if not args.out.is_dir():
            logging.error("Output has to be directory.")
            sys.exit(1)
    if not args.input.exists():
        logging.error("Input directory does not exist.")
        sys.exit(1)

    logging.info(f"Found {len(list(args.input.glob('*')))} books")
    return args


def main():
    args = get_args()
    model = NougatModel.from_pretrained(args.checkpoint)
    model = move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0)
    if args.batchsize <= 0:
        # set batch size to 1. Need to check if there are benefits for CPU conversion for >1
        args.batchsize = 1
    model.eval()
    png_paths = []
    for png_path in args.input.glob("*/*.png"):
        if args.out:
            out_path = args.out / png_path.parent / png_path.with_suffix(".mmd").name
            if out_path.exists() and not args.recompute:
                logging.info(
                    f"Skipping {str(out_path)}, already computed. Run with --recompute to convert again."
                )
            else:
                png_paths.append(png_path)
    dataset = LazyImageDataset(
        png_paths, partial(model.encoder.prepare_input, random_padding=False)
    )
    if len(dataset) == 0:
        return
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyImageDataset.ignore_none_collate,
        num_workers=1,
        prefetch_factor=2,
    )

    predictions = []
    for i, (sample, file_paths) in enumerate(tqdm(dataloader)):
        model_output = model.inference(
            image_tensors=sample, early_stopping=args.skipping
        )
        # check if model output is faulty
        for j, output in enumerate(model_output["predictions"]):
            if output.strip() == "[MISSING_PAGE_POST]":
                # uncaught repetitions -- most likely empty page
                predictions.append(f"")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    # If we end up here, it means the output is most likely not complete and was truncated.
                    # logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"")
                else:
                    # If we end up here, it means the document page is too different from the training domain.
                    # This can happen e.g. for cover pages.
                    predictions.append(f"")
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)

            if args.out:
                out = "".join(predictions).strip()
                out = re.sub(r"\n{3,}", "\n\n", out).strip()
                file_path = Path(file_paths[j])
                out_path = (
                    args.out
                    / file_path.parent.stem
                    / file_path.with_suffix(".mmd").name
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(out, encoding="utf-8")
            else:
                print(out, "\n\n")

            predictions = []
        torch.cuda.empty_cache()


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time taken: ", time.time() - start, "s")
