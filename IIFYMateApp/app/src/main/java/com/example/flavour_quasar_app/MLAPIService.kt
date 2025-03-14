package com.example.flavour_quasar_app


import android.content.Context
import android.net.Uri
import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class MLApiService(private val baseUrl: String = "http://10.0.2.2:8002") {
    private val client = OkHttpClient()

    suspend fun uploadImage(imageUri: Uri, context: Context): PredictedValues? {
        return try {
            // Read the image file
            val imageBytes = context.contentResolver.openInputStream(imageUri)?.readBytes()
                ?: throw IOException("Could not read image file")

            // Create request body
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    "meal.jpg",
                    imageBytes.toRequestBody("image/jpeg".toMediaType())
                )
                .build()

            // Create request
            val request = Request.Builder()
                .url("$baseUrl/estimate-calories/")
                .post(requestBody)
                .build()

            // Make the request
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
                val responseString = response.body?.string() ?: return null
                Log.d("MLApiService", "Response: $responseString")  // Add this for debugging
                val jsonResponse = JSONObject(responseString)

                return PredictedValues(
                    fat = jsonResponse.getInt("fat"),
                    carbs = jsonResponse.getInt("carbs"),
                    protein = jsonResponse.getInt("protein"),
                    calories = jsonResponse.getInt("calories")
                )
            } else {
                Log.e("MLApiService", "Error response: ${response.code} - ${response.body?.string()}")
                null
            }
        } catch (e: Exception) {
            Log.e("MLApiService", "Error uploading image", e)
            null
        }
    }
}


// Data class for the predictions
data class PredictedValues(
    val fat: Int,
    val carbs: Int,
    val protein: Int,
    val calories: Int
)