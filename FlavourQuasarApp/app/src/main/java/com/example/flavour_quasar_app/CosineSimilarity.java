package com.example.flavour_quasar_app;
/*
import com.google.auth.oauth2.GoogleCredentials;
import com.google.cloud.bigquery.BigQuery;
import com.google.cloud.bigquery.BigQueryOptions;
import com.google.cloud.bigquery.QueryJobConfiguration;
import com.google.cloud.bigquery.Job;
import com.google.cloud.bigquery.JobId;
import com.google.cloud.bigquery.JobInfo;
import com.google.cloud.bigquery.TableResult;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.UUID;

public class CosineSimilarity {
    public static void main(String[] args) {
        String gcpConfigFile = "../flavourquasar-gcp-key.json";
        try (FileInputStream inputStream = new FileInputStream(gcpConfigFile)) {
            GoogleCredentials credentials = GoogleCredentials.fromStream(inputStream);
            BigQueryOptions options = BigQueryOptions.newBuilder().setCredentials(credentials).build();
            BigQuery bigQuery = options.getService();
            String projectId = "flavourquasar";
            String query = "SELECT healthLabels, label, ingredientLines, totalNutrients FROM `flavourquasar.edamam_recipes.edamam_raw_data`";
            QueryJobConfiguration queryConfig = QueryJobConfiguration.newBuilder(query)
                    .setUseLegacySql(false) // Set to true if you're using legacy SQL syntax
                    .build();

            JobId jobId = JobId.of(UUID.randomUUID().toString());
            Job queryJob = bigQuery.create(JobInfo.newBuilder(queryConfig).setJobId(jobId).build());

            queryJob = queryJob.waitFor();

            if (queryJob != null && queryJob.getStatus() != null && queryJob.getStatus().getError() == null) {
                // Query job completed successfully, process the result
                TableResult result = queryJob.getQueryResults();
                // Process the result here
                System.out.println("Query completed successfully");
            } else {
                // Query job failed
                System.out.println("Error executing query: " + queryJob.getStatus().getError().getMessage());
            }
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
    public static String myMethod(String input) {
        return input;
    }
}
*/