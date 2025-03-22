package com.example.flavour_quasar_app


import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class MLApiService(private val baseUrl: String = "http://api.ravinderrai.com:8002") {
    private val client = OkHttpClient()

    suspend fun uploadImage(imageUri: Uri, context: Context): PredictedValues? {
        return try {
            // Normalize the image
            val bitmap = withContext(Dispatchers.IO) {
                context.contentResolver.openInputStream(imageUri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream)
                }
            } ?: throw IOException("Could not decode image from URI")

            Log.d("ImageDebug", "Bitmap created: $bitmap")
            Log.d("ImageDebug", "Bitmap config: ${bitmap.config}, Width: ${bitmap.width}, Height: ${bitmap.height}")

            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 1024, 768, true)
            Log.d("ImageDebug", "Resized bitmap: ${resizedBitmap.width}x${resizedBitmap.height}")

            val stream = ByteArrayOutputStream()
            resizedBitmap.compress(Bitmap.CompressFormat.JPEG, 90, stream)
            val normalizedImageBytes = stream.toByteArray()

            Log.d("ImageDebug", "Normalized JPEG byte size: ${normalizedImageBytes.size}")

            if (normalizedImageBytes.size > 2_000_000) { // 2MB
                Log.w("ImageDebug", "Image still too large after resize!")
            }

            // Read the image file
            //val imageBytes = context.contentResolver.openInputStream(imageUri)?.readBytes()
            //   ?: throw IOException("Could not read image file")

            // Create request body
            val requestBody = MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart(
                    "file",
                    "meal.jpg",
                    normalizedImageBytes.toRequestBody("image/jpeg".toMediaType())
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
                val responseString = withContext(Dispatchers.IO) {
                    response.body?.string()
                } ?: return null

                Log.d("MLApiService", "Response: $responseString")  // Add this for debugging
                val jsonResponse = JSONObject(responseString)

                return PredictedValues(
                    fat = jsonResponse.getInt("fat"),
                    carbs = jsonResponse.getInt("carbohydrates_net"),
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