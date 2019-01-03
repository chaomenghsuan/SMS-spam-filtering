# SMS-spam-filtering
content-based SMS spam filtering with machine learning models

data from [this Kaggle page](https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## How to Run
adjust `feat` and `inDir` to run `SMS_script.py` file

The order of structural features is: `['message length', 'numeric ratio', 'non_alphanumeric ratio', 'url existance', 'uppercase character ratio', 'terms count']`. If you want to keep all 6 structural features, set `feat = [True, True, True, True, True, True]`

Set `inDir` the directory of `spam.csv` file.
