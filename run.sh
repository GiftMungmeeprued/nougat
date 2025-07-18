#!/bin/bash
source /home/gift/miniconda3/bin/activate nougat
for pdf in /home/gift/projects/noetarium-textbook-search/data/books-pdf-quant-v3/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  python pdf2img.py --pdf "$pdf" --output "/home/gift/projects/noetarium-textbook-search/data/books-png-quant-v3/$filename" --format "png"
done

python predict_page.py /home/gift/projects/noetarium-textbook-search/data/books-png-quant-v3 -o /home/gift/projects/noetarium-textbook-search/data/books-md-quant-v3 --batchsize 6