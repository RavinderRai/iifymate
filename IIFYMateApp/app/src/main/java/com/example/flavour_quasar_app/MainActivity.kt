package com.example.flavour_quasar_app

import android.app.Activity
import android.app.Dialog
import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
//import android.util.Log
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject
import org.json.JSONArray


class MainActivity : ComponentActivity() {
    private lateinit var buttonOpenPopup: Button
    private lateinit var buttonPredictMacros: Button
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Testing git change with this comment
        val editText: EditText = findViewById<EditText>(R.id.input_recipe_name)

        var selectedDietType = ""
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

        // grey out - meaning disable - button to make predictions until user enters a recipe
        buttonOpenPopup = findViewById(R.id.edit_ingredients)
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

            val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

            coroutineScope.launch {
                try {
                    val predictedIngredients = predictor.getIngredientsList(userInputRecipe)
                    showPopupWindow(userInputRecipe, selectedDietType, predictedIngredients)
                } catch (e: Exception) {
                    // Handle exceptions, if any
                } finally {
                    loadingDialog.dismiss()
                }
            }
        }

        buttonPredictMacros.setOnClickListener() {
            val loadingDialog = showLoadingDialog()

            val userInputRecipe = editText.text.toString()

            val predictor = FlaskPredictor()

            val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

            var predictedFat = 0
            var predictedCarbs = 0
            var predictedProtein = 0
            var predictedCalories = 0
            var macroPredictionInput: String

            coroutineScope.launch {
                try {
                    val ingredientsList = predictor.getIngredientsList(userInputRecipe)
                    val concatenatedIngredients = ingredientsList.joinToString(separator = " ")
                    macroPredictionInput = "$selectedDietType $userInputRecipe $concatenatedIngredients"

                    val macroPredictions = predictor.getMacroPredictions(macroPredictionInput)

                    val (fat, carbs, protein, calories) = macroPredictions
                    predictedFat = fat
                    predictedCarbs = carbs
                    predictedProtein = protein
                    predictedCalories = calories

                    // Use the predicted values as needed
                    // For example, pass them to another function or update UI components
                } catch (e: Exception) {
                    // Handle exceptions, if any
                } finally {
                    loadingDialog.dismiss()
                    launchMacrosWindow(userInputRecipe, predictedCalories, predictedFat, predictedCarbs, predictedProtein)
                }
            }
        }
    }
    private fun showPopupWindow(userInput: String, selectedDietType: String, ingredientsList: List<String>) {
        val popupView = layoutInflater.inflate(R.layout.ingredients_popup, null)

        val popupWindow = PopupWindow(
            popupView,
            (resources.displayMetrics.widthPixels * 0.9).toInt(), // Set width to 90% of screen width
            (resources.displayMetrics.heightPixels * 0.85).toInt(), // Set height to 85% of screen height
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

            val scrollView: ScrollView = popupView.findViewById(R.id.ingredients_popup)
            scrollView.post {
                scrollView.fullScroll(ScrollView.FOCUS_DOWN)
            }
        }

        val buttonGetMacros = popupView.findViewById<Button>(R.id.buttonGetMacros)
        buttonGetMacros.setOnClickListener {
            val loadingDialog = showLoadingDialog()

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

            //Log.d("Model Prediction Input", macroPredictionInput)

            val predictor = FlaskPredictor()

            val coroutineScope = CoroutineScope(Dispatchers.Main + SupervisorJob())

            var predictedFat = 0
            var predictedCarbs = 0
            var predictedProtein = 0
            var predictedCalories = 0

            coroutineScope.launch {
                try {

                    val macroPredictions = predictor.getMacroPredictions(macroPredictionInput)

                    val (fat, carbs, protein, calories) = macroPredictions
                    predictedFat = fat
                    predictedCarbs = carbs
                    predictedProtein = protein
                    predictedCalories = calories
                } catch (e: Exception) {
                    // Handle exceptions, if any
                } finally {
                    loadingDialog.dismiss()
                    popupWindow.dismiss()
                    launchMacrosWindow(userInput, predictedCalories, predictedFat, predictedCarbs, predictedProtein)
                }
            }
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
    // http://192.168.0.165:5000 // local link
    class FlaskPredictor(private val baseUrl: String = "https://flask-pred-go4e5g7vpq-uc.a.run.app") {

        private val client = HttpClient()

        private suspend fun predict(endpoint: String, inputData: String): String? {
            val url = "$baseUrl/$endpoint"
            return try {

                val response: HttpResponse = client.post(url) {
                    val jsonInput = JSONObject().apply {
                        put("user_input", inputData)
                    }
                    // Set the JSON object as the body of the request
                    setBody(jsonInput.toString())
                }


                //Log.d("Response", response.toString())

                if (response.status == HttpStatusCode.OK) {
                    // Read the response body as a string
                    val responseBody = response.bodyAsText()

                    // Log the response body
                    //Log.d("ResponseBody", "Response body: $responseBody, Type: ${responseBody.javaClass.simpleName}")
                    responseBody
                } else {
                    //Log.e("Response", "Error: ${response.status}")
                    null
                }
            } catch (e: Exception) {
                //.e("Response", "Error occurred: ${e.message}")
                null
            }
        }
        suspend fun getMacroPredictions(macroPredictionInput: String): PredictedValues {
            var predictedFat = 0.0
            var predictedCarbs = 0.0
            var predictedProtein = 0.0
            var predictedCalories = 0.0

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
                //Log.d("PredictedValues", "Fat: $predictedFat, Carbs: $predictedCarbs, Protein: $predictedProtein, Calories: $predictedCalories")
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
            //Log.d("ingredientsType", "Response body: $ingredients, Type: ${ingredients.javaClass.simpleName}")
            return ingredients
        }
    }
    private fun showLoadingDialog(): Dialog {
        val dialog = Dialog(this)
        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE)
        dialog.setContentView(R.layout.loading_layout)
        dialog.setCancelable(false)
        dialog.window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        dialog.show()
        return dialog
    }
    private fun launchMacrosWindow(userInputRecipe: String, predictedCalories: Int, predictedFat: Int, predictedCarbs: Int, predictedProtein: Int) {
        val intent = Intent(this, MacrosDisplay::class.java)
        intent.putExtra("recipe_name", userInputRecipe)
        intent.putExtra("calories", predictedCalories)
        intent.putExtra("fat", predictedFat)
        intent.putExtra("carbs", predictedCarbs)
        intent.putExtra("protein", predictedProtein)
        startActivity(intent)
    }
}
