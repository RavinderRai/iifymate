# Testing ML Model

To create the test dataset of the raw data for unit testing regarding the ML model, start the local postgresql service in WSL:

```bash
sudo service postgresql start
```

and then run:

```bash
psql -U iifymate -d recipe_data -c "\copy (SELECT * FROM raw_recipes ORDER BY RANDOM() LIMIT 
100) TO 'ml_features/ml_calorie_estimation/tests/data/test_raw_data.csv' WITH CSV HEADER"
```

and to get a test dataset for the clean data:

```bash
psql -U iifymate -d recipe_data -c "\copy (SELECT * FROM clean_recipes ORDER BY RANDOM() LIMIT 
100) TO 'ml_features/ml_calorie_estimation/tests/data/test_clean_data.csv' WITH CSV HEADER"
```