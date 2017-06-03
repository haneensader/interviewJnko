# JoonTalk
This repository contains 3 files:
1. Jupyter nootebook: This goes through the stages of my work, step by step, including data analysis, data processing, training the model, and a few examples.
2. A pickle file: This includes a saved snapshot of the trained model
3. A python file: This includes simple code to load the model, and run a few examples.

Improvements to be made:
1. Female/Male balance:
  To maintain a balanced dataset I subsampled my data. This is NEVER ok, because I am losing data that might contain helpful information.
  A better approach is to train several trees on different balanced subsets of the data --> Manual forest.
2. Trying a different model:
  SVM can have potential here but will require additional processing of the continuos features.
