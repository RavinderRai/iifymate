package com.example.flavour_quasar_app

import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.os.PersistableBundle
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ListView
import android.widget.PopupWindow
import android.widget.ScrollView
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
    private lateinit var buttonOpenPopup: Button
    private lateinit var scrollView: ScrollView
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

        /*
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
         */

        buttonOpenPopup = findViewById(R.id.enter_button)
        scrollView = findViewById(R.id.scrollView)
        //var scrollView = findViewById(R.id.scrollView)
        buttonOpenPopup.setOnClickListener() {
            showPopupWindow()
            // val inflater = getSystemService(LAYOUT_INFLATER_SERVICE) as? LayoutInflater
            // inflater?.let { layoutInflater ->
            //    val popupView = layoutInflater.inflate(R.layout.ingredients_popup, null)
                //val itemList = mutableListOf("Item 1", "Item 2", "Item 3")
            //}
        }
    }
    private fun showPopupWindow() {
        val popupView = layoutInflater.inflate(R.layout.ingredients_popup, null)

        val popupWindow = PopupWindow(
            popupView,
            (resources.displayMetrics.widthPixels * 0.9).toInt(), // Set width to 80% of screen width
            (resources.displayMetrics.heightPixels * 0.85).toInt(), // Set height to 60% of screen height
            true
        )
        popupWindow.setBackgroundDrawable(ColorDrawable(Color.argb(255, 215, 250, 214)))

        val editTextContainer = popupView.findViewById<LinearLayout>(R.id.editTextContainer)

        val listOfItems = mutableListOf("Item 1", "Item 2", "Item 3")
        for (item in listOfItems) {
            val horizontalLayout = LinearLayout(this)
            horizontalLayout.orientation = LinearLayout.HORIZONTAL

            val editText = createEditText(item)
            //val editText = EditText(this)
            //editText.setText(item)

            horizontalLayout.addView(editText)

            val closeButton = createCloseButton()
            closeButton.setOnClickListener {
                editTextContainer.removeView(horizontalLayout)
                //editTextContainer.removeView(closeButton)
            }
            horizontalLayout.addView(closeButton)
            editTextContainer.addView(horizontalLayout)
        }

        val buttonAddEditText = popupView.findViewById<Button>(R.id.buttonAddEditText)
        buttonAddEditText.setOnClickListener {
            // Create a horizontal LinearLayout to hold EditText and close button
            val horizontalLayout = LinearLayout(this)
            horizontalLayout.orientation = LinearLayout.HORIZONTAL

            // Add EditText
            //val newEditText = EditText(this)
            //newEditText.setText("")
            val newEditText = createEditText("")
            horizontalLayout.addView(newEditText)

            // Add a close button next to the new EditText
            val closeButton = createCloseButton()
            closeButton.setOnClickListener {
                editTextContainer.removeView(horizontalLayout)
            }
            horizontalLayout.addView(closeButton)
            editTextContainer.addView(horizontalLayout)

            //newEditText.setOnClickListener {
            //    editTextContainer.removeView(newEditText)
            //}
        }

        val buttonGetMacros = popupView.findViewById<Button>(R.id.buttonGetMacros)
        buttonGetMacros.setOnClickListener {
            val placeHolderFat = 30
            val placeHolderCarbs = 45
            val placeHolderProtein = 25
            val calories = placeHolderProtein*4 + placeHolderCarbs*4 + placeHolderFat*9

            val intent = Intent(this, MacrosDisplay::class.java)
            // Pass data to the new activity if needed
            intent.putExtra("calories", calories)
            intent.putExtra("fat", placeHolderFat)
            intent.putExtra("carbs", placeHolderCarbs)
            intent.putExtra("protein", placeHolderProtein)
            startActivity(intent)

            popupWindow.dismiss()
        }

        // Show the popup window
        popupWindow.showAtLocation(buttonOpenPopup, Gravity.CENTER, 0, 0)
    }
    private fun createCloseButton(): ImageButton {
        val closeButton = ImageButton(this)
        closeButton.setImageResource(android.R.drawable.ic_input_delete)
        closeButton.setBackgroundColor(Color.argb(128, 64, 64, 64))
        //closeButton.setPadding(10, 10, 10, 10)
        val params = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.WRAP_CONTENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        params.gravity = Gravity.END or Gravity.CENTER_VERTICAL
        closeButton.layoutParams = params
        return closeButton
    }
    private fun createEditText(text: String): EditText {
        val editText = EditText(this)

        // Set text
        editText.setText(text)

        // Set width
        val params = LinearLayout.LayoutParams(
            0, // Set width to 0 to allow weight to determine width
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        params.setMargins(0, 0, 20, 0)
        params.weight = 1f // Set weight to 1 to make EditText occupy available space evenly
        editText.layoutParams = params

        return editText
    }

}
