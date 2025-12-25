# Calibration Report

## Search for Official Evaluator

### Paths Searched
- `/home/runner/work/Santa-2025/Santa-2025/` (entire repo)
- All `.py`, `.ipynb`, `.md` files

### Keywords Used
- evaluate, metric, score, normalized, bounding square, overlap
- tree polygon vertices, anchor, trunk, rotation convention
- official, kaggle, competition

### Results
**Official evaluator NOT found in repository.**

The repository contains:
- `sample_submission.csv` - Sample submission file
- `submission_best_165.csv` - Best submission with Kaggle score 165.22

The repository does NOT contain:
- Official competition evaluation code
- Official tree polygon definition
- Official geometry/collision detection code

## Calibration Attempt

### Local Score vs Kaggle Baseline
| Submission | Local Score | Kaggle Score | Delta |
|------------|-------------|--------------|-------|
| submission_best_165.csv | 134.897 | 165.224 | 30.327 |

### Analysis
The local solver uses a reverse-engineered tree geometry (isoceles triangle, base=height=0.30) that does NOT match the official competition geometry.

**Calibration FAILED**: Local score (134.897) does not match Kaggle baseline (165.224) within tolerance (1e-3).

### Conclusion
Without the official evaluator code in this repository, accurate calibration is impossible. The optimization would be meaningless as improvements to the local score would not translate to improvements on the Kaggle leaderboard.

## Recommendation
To proceed with meaningful optimization:
1. Add the official competition evaluation code to the repository
2. Add the official tree polygon definition
3. Re-run calibration to verify local score matches Kaggle score
