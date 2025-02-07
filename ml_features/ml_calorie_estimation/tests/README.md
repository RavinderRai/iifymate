# Testing ML Model

To create the test dataset for the ML model, start the local postgresql service in WSL:

```bash
sudo service postgresql restart
```

and then run:

```bash
psql -U iifymate -d recipe_data -c "\copy (SELECT * FROM raw_recipes ORDER BY RANDOM() LIMIT 
100) TO 'ml_features/ml_calorie_estimation/tests/data/test_raw_data.csv' WITH CSV HEADER"
```