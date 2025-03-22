# Macro Estimation Model

## Notes

### MLFlow Setup

Once a new RDS instance is created, run

```
aws rds describe-db-instances --query 'DBInstances[*].MasterUsername'
```

to get the username. Then run 

```
psql -h iifymate-db.co5im862y9q7.us-east-1.rds.amazonaws.com -U username -d postgres
```

to connect to the RDS instance. Then run

```
CREATE DATABASE mlflowdb;
```

to create the mlflow database. Then run

```
GRANT ALL PRIVILEGES ON DATABASE mlflowdb TO iifymateadmin;
```

to grant all privileges to the mlflow database to the iifymateadmin user.

Reference: https://www.restack.io/docs/mlflow-knowledge-mlflow-aws-rds-integration