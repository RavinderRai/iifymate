package com.example.flavour_quasar_app

import android.app.Activity
import android.app.Dialog
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.Window
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.ScrollView
import android.widget.Spinner
import androidx.activity.ComponentActivity
import io.ktor.client.*
//import com.example.flavour_quasar_app.CosineSimilarity
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.client.statement.bodyAsText
import io.ktor.http.HttpStatusCode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.json.JSONObject
import org.json.JSONArray


class MainActivity : ComponentActivity() {
    private lateinit var buttonOpenPopup: Button
    private lateinit var buttonPredictMacros: Button
    private lateinit var scrollView: ScrollView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val editText: EditText = findViewById<EditText>(R.id.input_recipe_name)
        //val userInput = editText.text.toString()

        var selectedDietType: String = ""
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
                selectedDietType = parent?.getItemAtPosition(position).toString()
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {
                selectedDietType = "Balanced"
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

        // grey out - meaning disable - button to make predictions until user enters a recipe
        buttonOpenPopup = findViewById(R.id.enter_button)
        buttonPredictMacros = findViewById(R.id.get_macros)
        buttonOpenPopup.isEnabled = false
        buttonPredictMacros.isEnabled = false

        // text change listener to enable buttons when recipe is entered
        editText.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) {
                // Enable or disable the button based on whether EditText is empty
                buttonOpenPopup.isEnabled = !s.isNullOrBlank()
                buttonPredictMacros.isEnabled = !s.isNullOrBlank()
            }
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {
                // No implementation needed
            }
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                // No implementation needed
            }
        })
        buttonOpenPopup.setOnClickListener() {
            KeyboardUtils.hideKeyboard(this)

            val loadingDialog = showLoadingDialog()

            val userInputRecipe = editText.text.toString()


            val predictor = FlaskPredictor()

            val predictedIngredients = runBlocking {
                predictor.getIngredientsList(userInputRecipe)
            }
            loadingDialog.dismiss()

            /*
            val ingredients = mutableListOf<String>()
            runBlocking {
                val ingredientsPrediction = predictor.predict("predict_ingredients", userInputRecipe)

                val jsonArray = JSONArray(ingredientsPrediction)
                for (i in 0 until jsonArray.length()) {
                    ingredients.add(jsonArray.getString(i))
                }
                Log.d("ingredientsType", "Response body: $ingredients, Type: ${ingredients.javaClass.simpleName}")
            }

             */
            showPopupWindow(userInputRecipe, selectedDietType, predictedIngredients)
        }

        buttonPredictMacros.setOnClickListener() {
            val loadingDialog = showLoadingDialog()

            val userInputRecipe = editText.text.toString()

            val predictor = FlaskPredictor()

            val macroPredictions = runBlocking {
                val ingredientsList = predictor.getIngredientsList(userInputRecipe)
                val concatenatedIngredients = ingredientsList.joinToString(separator = " ")
                val macroPredictionInput = "$selectedDietType $userInputRecipe $concatenatedIngredients"

                predictor.getMacroPredictions(macroPredictionInput)
            }
            val (predictedFat, predictedCarbs, predictedProtein, predictedCalories) = macroPredictions

            val intent = Intent(this, MacrosDisplay::class.java)
            // Pass data to the new activity if needed
            intent.putExtra("recipe_name", userInputRecipe)
            intent.putExtra("calories", predictedCalories)
            intent.putExtra("fat", predictedFat)
            intent.putExtra("carbs", predictedCarbs)
            intent.putExtra("protein", predictedProtein)

            loadingDialog.dismiss()

            startActivity(intent)

        }
    }
    private fun showPopupWindow(userInput: String, selectedDietType: String, ingredientsList: List<String>) {
        val popupView = layoutInflater.inflate(R.layout.ingredients_popup, null)

        val popupWindow = PopupWindow(
            popupView,
            (resources.displayMetrics.widthPixels * 0.9).toInt(), // Set width to 80% of screen width
            (resources.displayMetrics.heightPixels * 0.85).toInt(), // Set height to 60% of screen height
            true
        )

        // dismiss keyboard if user clicks outside popup
        val rootView = findViewById<View>(android.R.id.content)
        KeyboardUtils.setupKeyboardCloseOnTouch(this, rootView)

        popupWindow.setBackgroundDrawable(ColorDrawable(Color.argb(255, 215, 250, 214)))

        val editTextContainer = popupView.findViewById<LinearLayout>(R.id.editTextContainer)

        // val ingredientsList = mutableListOf("Item 1", "Item 2", "Item 3")
        for (item in ingredientsList) {
            val horizontalLayout = LinearLayout(this)
            horizontalLayout.orientation = LinearLayout.HORIZONTAL

            val editText = createEditText(item)

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
            val newEditText = createEditText("")
            horizontalLayout.addView(newEditText)

            // Add a close button next to the new EditText
            val closeButton = createCloseButton()
            closeButton.setOnClickListener {
                editTextContainer.removeView(horizontalLayout)
            }
            horizontalLayout.addView(closeButton)
            editTextContainer.addView(horizontalLayout)

        }

        val buttonGetMacros = popupView.findViewById<Button>(R.id.buttonGetMacros)
        buttonGetMacros.setOnClickListener {

            // get all editText items (ingredients) and concatenate them
            val concatenatedIngredients = StringBuilder()
            for (i in 0 until editTextContainer.childCount) {
                val horizontalLayout = editTextContainer.getChildAt(i) as? LinearLayout
                // Iterate through each child view (EditText) in horizontalLayout
                horizontalLayout?.let {
                    for (j in 0 until it.childCount) {
                        val view = it.getChildAt(j)
                        if (view is EditText) {
                            // Append the text from each EditText to the concatenatedText
                            concatenatedIngredients.append(view.text.toString())
                            concatenatedIngredients.append(" ") // Add space separator between EditText values
                        }
                    }
                }
            }
            val allEditTextValues = concatenatedIngredients.toString()
            val macroPredictionInput = "$selectedDietType $userInput $allEditTextValues"

            Log.d("Model Prediction Input", macroPredictionInput)

            val predictor = FlaskPredictor()

            val macroPredictions = runBlocking {
                predictor.getMacroPredictions(macroPredictionInput)
            }
            val (predictedFat, predictedCarbs, predictedProtein, predictedCalories) = macroPredictions

            //runBlocking {
                /*
                val macrosPrediction = predictor.predict("predict_macros", macroPredictionInput)

                macrosPrediction?.let { response ->
                    // Parse JSON response
                    val jsonResponse = JSONObject(response)

                    // Extract predicted values
                    predictedFat = jsonResponse.getDouble("predicted_fat")
                    predictedCarbs = jsonResponse.getDouble("predicted_carbs")
                    predictedProtein = jsonResponse.getDouble("predicted_protein")
                    predictedCalories = jsonResponse.getDouble("calories")



                    // Now you have the predicted values, you can use them as needed
                    Log.d("PredictedValues", "Fat: $predictedFat, Carbs: $predictedCarbs, Protein: $predictedProtein, Calories: $predictedCalories")
                }

                 */
            //}

            val intent = Intent(this, MacrosDisplay::class.java)
            // Pass data to the new activity if needed
            intent.putExtra("recipe_name", userInput)
            intent.putExtra("calories", predictedCalories.toInt())
            intent.putExtra("fat", predictedFat.toInt())
            intent.putExtra("carbs", predictedCarbs.toInt())
            intent.putExtra("protein", predictedProtein.toInt())
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
    class KeyboardUtils{

        companion object{
            fun hideKeyboard(activity: Activity) {
                val imm: InputMethodManager = activity.getSystemService(Activity.INPUT_METHOD_SERVICE) as InputMethodManager
                var view: View? = activity.currentFocus
                if (view == null) {
                    view = View(activity)
                }
                imm.hideSoftInputFromWindow(view.windowToken, 0)
            }
            fun setupKeyboardCloseOnTouch(activity: Activity, rootView: View) {

                rootView.setOnTouchListener { _, event ->
                    if (event.action == MotionEvent.ACTION_DOWN) {
                        hideKeyboardOnTouchOutside(activity, rootView, event)
                        rootView.performClick()
                    }
                    false
                }
            }
            private fun hideKeyboardOnTouchOutside(activity: Activity, rootView: View, event: MotionEvent) {
                if (rootView.isFocusable && event.y < rootView.top) {
                    hideKeyboard(activity)
                }
            }
        }
    }
    data class PredictedValues(val fat: Int, val carbs: Int, val protein: Int, val calories: Int)
    class FlaskPredictor(private val baseUrl: String = "http://192.168.0.165:5000") {

        private val client = HttpClient()

        suspend fun predict(endpoint: String, inputData: String): String? {
            val url = "$baseUrl/$endpoint"
            return try {
                val response: HttpResponse = client.post(url) {
                    val jsonInput = JSONObject().apply {
                        put("user_input", inputData)
                    }
                    // Set the JSON object as the body of the request
                    setBody(jsonInput.toString())
                }
                Log.d("Response", response.toString())

                if (response.status == HttpStatusCode.OK) {
                    // Read the response body as a string
                    val responseBody = response.bodyAsText()

                    // Log the response body
                    Log.d("ResponseBody", "Response body: $responseBody, Type: ${responseBody.javaClass.simpleName}")
                    responseBody
                } else {
                    Log.e("Response", "Error: ${response.status}")
                    null
                }
            } catch (e: Exception) {
                Log.e("Response", "Error occurred: ${e.message}")
                null
            }
        }
        suspend fun getMacroPredictions(macroPredictionInput: String): PredictedValues {
            var predictedFat: Double = 0.0
            var predictedCarbs: Double = 0.0
            var predictedProtein: Double = 0.0
            var predictedCalories: Double = 0.0

            val macrosPrediction = predict("predict_macros", macroPredictionInput)

            macrosPrediction?.let { response ->
                // Parse JSON response
                val jsonResponse = JSONObject(response)

                // Extract predicted values
                predictedFat = jsonResponse.getDouble("predicted_fat")
                predictedCarbs = jsonResponse.getDouble("predicted_carbs")
                predictedProtein = jsonResponse.getDouble("predicted_protein")
                predictedCalories = jsonResponse.getDouble("calories")


                // Now you have the predicted values
                Log.d("PredictedValues", "Fat: $predictedFat, Carbs: $predictedCarbs, Protein: $predictedProtein, Calories: $predictedCalories")
            }

            // Return a Quad containing the predicted values and calories
            return PredictedValues(predictedFat.toInt(), predictedCarbs.toInt(), predictedProtein.toInt(), predictedCalories.toInt())
        }
        suspend fun getIngredientsList(inputRecipeName: String): List<String> {
            val ingredientsPrediction = predict("predict_ingredients", inputRecipeName)

            val ingredients = mutableListOf<String>()

            val jsonArray = JSONArray(ingredientsPrediction)
            for (i in 0 until jsonArray.length()) {
                ingredients.add(jsonArray.getString(i))
            }
            Log.d("ingredientsType", "Response body: $ingredients, Type: ${ingredients.javaClass.simpleName}")
            return ingredients
        }
    }
    fun showLoadingDialog(): Dialog {
        val dialog = Dialog(this)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(R.layout.loading_layout)
        dialog.setCancelable(false)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.show()
        return dialog
    }
}
