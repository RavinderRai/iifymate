package com.example.flavour_quasar_app

import android.os.Bundle
import android.os.PersistableBundle
import android.view.LayoutInflater
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.Spinner
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
import com.example.flavour_quasar_app.ui.theme.Flavour_Quasar_AppTheme
import com.example.flavour_quasar_app.CosineSimilarity
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

        val url = "http://127.0.0.1:5000/predict_ingredients" // Assuming your Flask app is running on the same machine as the Android emulator or device

        val userInputRecipe = "Black Bean Tacos"

        val client = HttpClient(CIO)
        

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
