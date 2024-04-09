package com.example.flavour_quasar_app

import android.os.Bundle
import android.os.PersistableBundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.lifecycleScope
import com.example.flavour_quasar_app.ui.theme.Flavour_Quasar_AppTheme
//import com.example.flavour_quasar_app.CosineSimilarity
import io.ktor.client.request.post
import io.ktor.client.request.request
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.client.statement.readText
import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONObject

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val editText: EditText = findViewById<EditText>(R.id.input_recipe_name)
        val userInput = editText.text.toString()

        val spinner: Spinner = findViewById<Spinner>(R.id.spinner)
        val adapter: ArrayAdapter<CharSequence> = ArrayAdapter.createFromResource(
            this,
            R.array.dropdown_items,
            android.R.layout.simple_spinner_item
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinner.adapter = adapter

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val selectedItem = parent?.getItemAtPosition(position).toString()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {
                val defaultSelectedItem = "Balanced"
            }

        }


        //val url = "http://127.0.0.1:5000/predict_ingredients"
        val url = "http://192.168.0.165:5000/predict_ingredients"
        val userInputRecipe: String = "Black Bean Tacos"

        runBlocking {
            try {
                val client = HttpClient()
                val response: HttpResponse = client.post(url) {
                    val jsonInput = JSONObject().apply {
                        put("user_input", userInputRecipe)
                    }
                    // Set the JSON object as the body of the request
                    setBody(jsonInput.toString())
                }
                Log.d("Response", response.toString())

                if (response.status == HttpStatusCode.OK) {
                    // Read the response body as a string
                    val responseBody = response.bodyAsText()
                    // Log the response body
                    Log.d("Response", "Response body: $responseBody")
                } else {
                    Log.e("Response", "Error: ${response.status}")
                }
            } catch (e: Exception) {
                Log.e("Response", "Error occurred: ${e.message}")
            }
        }

        val button: Button = findViewById(R.id.enter_button)
        button.setOnClickListener() {
            val inflater = getSystemService(LAYOUT_INFLATER_SERVICE) as? LayoutInflater
            inflater?.let { layoutInflater ->
                val popupView = layoutInflater.inflate(R.layout.ingredients_popup, null)

                val popupWindow = PopupWindow(
                    popupView,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )

                popupWindow.showAsDropDown(button)
            }
        }
    }
}
