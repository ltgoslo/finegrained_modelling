import numpy as np

original = np.array([[63.5, 57.3, 57.6, 84.3, 54.6, 74.1, 72.8, 48.8],
                     [64.3, 57.8, 58.7, 84.4, 55.6, 75.6, 74.3, 46.6],
                     [63.5, 57.3, 60.2, 84.4, 56.8, 74.1, 72.8, 46.1],
                     [60.8, 58.2, 57.8, 81.4, 61.4, 73.9, 74.5, 49.0],
                     [59.3, 57.8, 55.2, 81.3, 60.2, 77.2, 74.5, 48.5]])

mpqa = np.array([[60.3, 57.0, 61.3, 0.0, 57.5, 74.2, 72.2, 0.0],
                 [61.6, 57.0, 59.8, 0.0, 55.2, 77.1, 73.9, 0.0],
                 [60.3, 57.0, 61.3, 0.0, 57.5, 74.2, 72.2, 0.0],
                 [59.1, 58.1, 57.0, 0.0, 63.7, 75.0, 74.7, 0.0],
                 [56.2, 58.1, 52.7, 0.0, 61.9, 75.6, 75.2, 0.0]])


dsservices = np.array([[63.6, 56.3, 60.6, 0.0, 53.4, 72.1, 72.2, 49.4],
                       [61.3, 54.5, 59.4, 0.0, 56.2, 76.3, 74.8, 45.4],
                       [64.1, 56.0, 60.1, 82.4, 56.1, 73.0, 72.4, 49.8],
                       [61.4, 56.7, 55.2, 0.0, 63.0, 76.5, 74.5, 48.1],
                       [58.5, 57.1, 54.8, 0.0, 59.0, 75.9, 73.2, 49.5]])

dsunis = np.array([[63.1, 57.0, 60.3, 0.0, 54.7, 73.6, 72.8, 0.0],
                   [64.1, 56.9, 58.2, 0.0, 55.2, 70.8, 72.9, 0.0],
                   [62.3, 57.0, 59.7, 0.0, 54.7, 73.6, 72.8, 0.0],
                   [59.6, 57.8, 53.4, 0.0, 62.0, 74.8, 71.0, 0.0],
                   [59.8, 57.5, 52.3, 0.0, 59.9, 74.8, 74.2, 0.0]])

huliu = np.array([[60.3, 56.2, 60.8, 82.8, 54.0, 73.6, 73.2, 50.6],
                  [61.2, 55.0, 61.2, 82.3, 46.5, 74.8, 74.2, 43.8],
                  [60.3, 56.2, 60.8, 82.8, 54.0, 73.6, 73.2, 50.6],
                  [59.7, 57.2, 56.4, 81.0, 61.1, 75.5, 73.7, 47.2],
                  [60.8, 57.1, 55.2, 80.8, 61.3, 73.8, 73.9, 49.2]])

nrc = np.array([[64.0, 56.9, 63.0, 83.1, 54.8, 72.0, 73.0, 49.4],
                [63.7, 56.9, 61.1, 83.3, 49.1, 74.9, 74.9, 46.1],
                [64.0, 56.9, 63.0, 83.1, 54.8, 72.0, 73.0, 49.4],
                [61.1, 58.0, 55.6, 80.4, 62.0, 75.3, 74.6, 49.7],
                [59.5, 57.6, 56.9, 80.8, 61.3, 75.4, 74.8, 49.8]])

socal = np.array([[63.2, 56.6, 60.5, 0.0, 51.5, 69.8, 71.0, 50.0],
                  [61.8, 53.7, 59.9, 0.0, 51.4, 72.8, 73.0, 45.4],
                  [63.2, 56.6, 60.5, 0.0, 51.5, 69.8, 71.0, 50.0],
                  [59.2, 57.8, 54.5, 0.0, 62.3, 71.5, 71.8, 49.5],
                  [59.7, 56.5, 55.6, 0.0, 60.9, 73.4, 73.0, 51.4]])

socalg = np.array([[62.6, 56.5, 60.0, 0.0, 53.2, 71.5, 72.9, 0.0],
                   [62.1, 56.2, 60.8, 0.0, 49.7, 74.9, 74.0, 0.0],
                   [62.6, 56.5, 60.0, 0.0, 53.2, 71.5, 72.9, 0.0],
                   [60.0, 57.8, 55.5, 0.0, 61.5, 74.8, 74.5, 0.0],
                   [60.6, 57.1, 54.6, 0.0, 60.5, 73.5, 72.7, 0.0]])

perms = [mpqa, dsservices, dsunis, huliu, nrc, socal, socalg]

diffs = []
for p in perms:
    diffs.append((p - original).mean(axis=0))

diffs = np.array(diffs)