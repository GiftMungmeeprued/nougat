#!/bin/bash

# source /home/gift/miniconda3/bin/activate nougat
# for pdf in /home/gift/projects/noetarium-textbook-search/data/webs-pdf-v1/*.pdf; do
#   filename=$(basename "$pdf" .pdf)
#   python pdf2img.py --pdf "$pdf" --output "/home/gift/projects/noetarium-textbook-search/data/webs-png-v1/$filename" --format "png"
# done

# source /home/gift/miniconda3/bin/activate nougat
# for pdf in /home/gift/projects/noetarium-textbook-search/data/books-pdf-v2/*.pdf; do
#   filename=$(basename "$pdf" .pdf)
#   python pdf2img.py --pdf "$pdf" --output "/home/gift/projects/noetarium-textbook-search/data/books-png-v2/$filename" --format "png"
# done

source /home/gift/miniconda3/bin/activate nougat
for pdf in /home/gift/projects/noetarium-textbook-search/data/books-pdf-v3/*.pdf; do
  filename=$(basename "$pdf" .pdf)
  python pdf2img.py --pdf "$pdf" --output "/home/gift/projects/noetarium-textbook-search/data/books-png-v3/$filename" --format "png"
done

python predict_page.py /home/gift/projects/noetarium-textbook-search/data/books-png-v3 -o /home/gift/projects/noetarium-textbook-search/data/books-md-v3 --batchsize 6