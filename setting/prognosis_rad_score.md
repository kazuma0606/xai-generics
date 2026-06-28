# Prognosis / Rad Score

Source notebook:

- `ipynb/rad_score_T2_to_FLAIR.ipynb`

Role:

- downstream prognosis-oriented analysis notebook
- works on radiomics feature tables and survival labels

Observed inputs:

- `real_FLAIR.csv`
- `fake_T2_to_FLAIR.csv`
- `days2.csv`

Observed concepts in notebook evidence:

- radiomics feature comparison between real and generated data
- radiomics score construction
- survival-related columns such as:
  - `day`
  - `arrest`
  - `name`

Observed downstream chain from project discussion and notebook evidence:

1. image translation
2. radiomics extraction
3. feature selection / radiomics score
4. Lasso-Cox style prognosis modeling
5. Logrank-based group comparison

Notes:

- the notebook filename is FLAIR-oriented, but the project discussion also ties
  the broader prognosis workflow to T2-side radiomics
- implementation should keep this stage conceptually separated from the first
  translation baseline
