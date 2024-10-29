# Amazon ML Challenge 2024 - Entity Value Extraction for E-Commerce Optimization

### Team Members: Paakhi Maheshwari, Aryaman Gupta, Animesh Seth

## Project Overview

This project, developed for the Amazon ML Challenge 2024, addresses the challenge of **automating entity value extraction** from raw image data, a crucial task for enhancing e-commerce experiences. Our solution achieved a **Top 100 ranking** out of over 74,000 participants, focusing on combining fast text recognition with high-accuracy vision-language modeling.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
  - [1. Text Recognition & Regular Expression](#1-text-recognition--regular-expression)
  - [2. Fine-Tuned Vision-Language Model](#2-fine-tuned-vision-language-model)
  - [3. Hybrid Approach](#3-hybrid-approach)
- [Implementation Details](#implementation-details)
  - [Tools Used](#requirements)

## Problem Statement

E-commerce platforms often rely on efficient entity value extraction from product images to support accurate listings, descriptions, and recommendations. Our objective was to build a robust, scalable model that automates this process with both high speed and accuracy, helping to reduce manual data entry and improve customer experiences.

## Solution Approach

### 1. Text Recognition & Regular Expression

Using **PaddleOCR**, we performed initial text extraction from images. We then applied a **Regular Expression (RegEx) algorithm** to identify and categorize entity values. This approach yielded high processing speed but lacked sufficient accuracy for some complex image data.

### 2. Fine-Tuned Vision-Language Model

To enhance accuracy, we fine-tuned **MiniCPM-v2.6**, a vision-language model. By tailoring specific prompts for entity value extraction, we improved recognition precision. This method achieved superior accuracy, though it was computationally intensive and slower compared to OCR and RegEx.

### 3. Hybrid Approach

To balance efficiency and precision, we combined the two methods:
- **PaddleOCR + RegEx** for fast, preliminary text extraction.
- **MiniCPM-v2.6** for a more refined and accurate recognition process.

This **hybrid approach** optimized our solution by leveraging the strengths of both techniques, achieving a notable balance between speed and accuracy.

## Implementation Details

### Tools Used

- Python 3.8+
- PaddleOCR
- MiniCPM-v2.6
- Regular Expressions (RegEx) library
