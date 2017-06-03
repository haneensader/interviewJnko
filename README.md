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

4. Features I did not use:
- I did not use the subject of the talk as a feature, or used it to analyze the data at all. However, the subject can have a strong effect on the number of people attending the talk.
- One more thing I didn't use was the name of the speaker itself, though it is known that speakers' reputation affect their ability to attract audience.

5. More data processing:
While I worked on the data I came across many rows with columns without labels, such as rows with missing gender. To make things easier on me, I simply dropped these rows. However, work can be done to fill the gaps in a way that more data can be used in our training. One wierd phenomena that I found, which can easily be solved, is that sometimes for the same speaker we find rows with missing gender, and rows with defined gender.

6. Gender vs. Biological Sex: I find it important to say that while I, and the dataset, used gender to refer to "female" and "male", these two labels do not describe gender, but biological sex :)
