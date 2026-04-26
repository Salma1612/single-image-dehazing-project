# Single Image Dehazing using Classical Image Processing

## Overview
This project implements a baseline haze removal algorithm and an improved dehazing method using OpenCV and NumPy.

## Features
- Dark Channel Prior based dehazing
- Atmospheric light estimation
- Transmission recovery
- Improved gamma correction
- Contrast enhancement
- PSNR / SSIM evaluation

## Results

| Method | PSNR | SSIM |
|-------|------|------|
| Baseline | 10.77 | 0.5768 |
| Improved | 16.41 | 0.8708 |

## Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib
- scikit-image

## Run

pip install -r requirements.txt

python improved_code.py

## Output
See /results folder.

## Author
Salma
B.Tech CSE (AI & ML)
