# JoonTalk
This repository contains 3 files:
1. Jupyter nootebook: This goes through the stages of my work, step by step, including data analysis, data processing, training the model, and a few examples.
2. A pickle file: This includes a saved snapshot of the trained model
3. A python file: This includes simple code to load the model, and run a few examples.

Notes:
1. Female/Male balance:
To maintain a balanced dataset I subsampled my data. This is NEVER ok, because I am losing data that might contain helpful information.
A better approach is to train several trees on different balanced subsets of the data --> a type of manual forest.

2. SVM:
I believe that the SVM  model can have potential here but will require additional processing of the continuos features.

3. NPS Score as a feature:
At first I started with a feature list that did not contain "NPS Score". The motivation was that "nps score" is merely an outcome and does not represent a "feature" of the talk. However, it turned out that adding the "nps score" as a feature improves the prediction for the number of attendees.
This might be due to cases where some talks' quality, which is represented by the "nps score", are pre known and people tend to attend these talks less or more depending on whether it is known to be good or bad.
