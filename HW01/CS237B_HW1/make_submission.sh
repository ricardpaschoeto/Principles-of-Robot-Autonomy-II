#!/usr/bin/env bash
zip -r hw1.zip Problem_1 Problem_2 \
  -x \
  "Problem_1/*.pkl" "Problem_1/__pycache__/*" \
  "Problem_2/trained_models/*" "Problem_2/retrain_logs/*" \
  "Problem_2/datasets/*" "Problem_2/__pycache__/*" \
  "*.npz" "*.npy"
