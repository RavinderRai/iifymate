package com.example.flavour_quasar_app
/*
import com.google.auth.oauth2.GoogleCredentials
import com.google.cloud.bigquery.BigQueryOptions
import com.google.cloud.bigquery.QueryJobConfiguration
import java.io.FileInputStream

fun main() {
    val gcpConfigFile = "../flavourquasar-gcp-key.json"
    val inputStream = FileInputStream(gcpConfigFile)
    val credentials = GoogleCredentials.fromStream(inputStream)
    val options = BigQueryOptions.newBuilder().setCredentials(credentials).build()

    val bigQuery = options.service
    val projectId = "flavourquasar"

    val query = """
        SELECT healthLabels, label, ingredientLines, totalNutrients
        FROM `flavourquasar.edamam_recipes.edamam_raw_data`
    """.trimIndent()

    val queryConfig = QueryJobConfiguration.newBuilder(query)
        .setUseLegacySql(false) // Set to true if you're using legacy SQL syntax
        .build()

    // Execute the query
    val queryJob = bigQuery.query(queryConfig)

    // Wait for the query to complete


}
 */