package com.example.flavour_quasar_app

import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class MacroPredictor(private val baseUrl: String = "http://localhost:8000") {
    private val client = OkHttpClient()
    private val JSON = "application/json; charset=utf-8".toMediaType()

    suspend fun predictMacros(text: String): PredictedValues {
        try {
            val jsonInput = JSONObject().apply {
                put("text", text)  // Changed from user_input to text to match FastAPI model
            }

            val requestBody = jsonInput.toString().toRequestBody(JSON)

            val request = Request.Builder()
                .url("$baseUrl/predict")
                .post(requestBody)
                .build()

            val response = suspendCoroutine { continuation ->
                client.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        continuation.resumeWithException(e)
                    }

                    override fun onResponse(call: Call, response: Response) {
                        continuation.resume(response)
                    }
                })
            }

            if (response.isSuccessful) {
                val responseString = response.body?.string() ?: throw IOException("Empty response body")
                val jsonResponse = JSONObject(responseString)

                return PredictedValues(
                    fat = jsonResponse.getDouble("fat").toInt(),
                    carbs = jsonResponse.getDouble("carbs").toInt(),
                    protein = jsonResponse.getDouble("protein").toInt(),
                    calories = jsonResponse.getDouble("calories").toInt()
                )
            } else {
                throw IOException("HTTP ${response.code}: ${response.message}")
            }
        } catch (e: Exception) {
            // Log error and return default values
            e.printStackTrace()
            return PredictedValues(0, 0, 0, 0)
        }
    }

    suspend fun batchPredictMacros(texts: List<String>): List<PredictedValues> {
        try {
            val jsonInput = JSONObject().apply {
                put("texts", texts)
            }

            val requestBody = jsonInput.toString().toRequestBody(JSON)

            val request = Request.Builder()
                .url("$baseUrl/batch-predict")
                .post(requestBody)
                .build()

            val response = suspendCoroutine { continuation ->
                client.newCall(request).enqueue(object : Callback {
                    override fun onFailure(call: Call, e: IOException) {
                        continuation.resumeWithException(e)
                    }

                    override fun onResponse(call: Call, response: Response) {
                        continuation.resume(response)
                    }
                })
            }

            if (response.isSuccessful) {
                val responseString = response.body?.string() ?: throw IOException("Empty response body")
                val jsonArray = JSONObject(responseString).getJSONArray("result")

                return List(jsonArray.length()) { i ->
                    val prediction = jsonArray.getJSONObject(i)
                    PredictedValues(
                        fat = prediction.getDouble("fat").toInt(),
                        carbs = prediction.getDouble("carbs").toInt(),
                        protein = prediction.getDouble("protein").toInt(),
                        calories = prediction.getDouble("calories").toInt()
                    )
                }
            } else {
                throw IOException("HTTP ${response.code}: ${response.message}")
            }
        } catch (e: Exception) {
            e.printStackTrace()
            return emptyList()
        }
    }
}