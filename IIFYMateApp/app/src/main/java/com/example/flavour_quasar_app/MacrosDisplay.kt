package com.example.flavour_quasar_app

import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
//import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.core.content.ContextCompat
import com.github.mikephil.charting.animation.Easing
import com.github.mikephil.charting.charts.PieChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.PieData
import com.github.mikephil.charting.data.PieDataSet
import com.github.mikephil.charting.data.PieEntry
import com.github.mikephil.charting.formatter.ValueFormatter
import com.github.mikephil.charting.utils.MPPointF
import java.util.Locale

class MacrosDisplay : ComponentActivity() {
    private lateinit var pieChart: PieChart
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.macros_display)

        // Retrieve data passed from previous activity
        val recipeName = intent.getStringExtra("recipe_name")
        val calories = intent.getIntExtra("calories", 0)
        val fat = intent.getIntExtra("fat", 0)
        val protein = intent.getIntExtra("protein", 0)
        val carbs = intent.getIntExtra("carbs", 0)

        val capitalizedRecipeName = recipeName?.capitalizeWords()

        //setting up a user alert in case model fails for whatever reason
        if (calories == 0 && fat == 0 && protein == 0 && carbs == 0) {
            // Display a Toast message to the user
            Toast.makeText(
                this,
                "Model request failed. Please try again.",
                Toast.LENGTH_LONG
            ).show()

            finish()
        }

        val fatRatio = (fat.toFloat() * 9 / calories.toFloat()) * 100
        val proteinRatio = (protein.toFloat() * 4 / calories.toFloat()) * 100
        val carbsRatio = (carbs.toFloat() * 4 / calories.toFloat()) * 100

        val textViewRecipeName = findViewById<TextView>(R.id.recipeName)

        //Log.d("RecipeName", "Recipe Name: $capitalizedRecipeName")
        if (recipeName != null) {
            textViewRecipeName.text = capitalizedRecipeName
            //setTextViewText(R.id.recipeName, recipeName)
        } else {
            //Log.d("RecipeName", "Recipe Name is null")
            textViewRecipeName.setText(R.string.no_recipe_name_input)
            //setTextViewText(R.id.recipeName, "No Recipe Input")
        }

        setTextViewText(R.id.calorieCount, "$calories")
        setTextViewText(R.id.fatCount, "$fat")
        setTextViewText(R.id.proteinCount, "$protein")
        setTextViewText(R.id.carbsCount, "$carbs")

        val returnButton = findViewById<Button>(R.id.returnActivityMain)
        returnButton.setOnClickListener {
            finish() // Finish the current activity and return to the previous activity (MainActivity)
        }

        pieChart = findViewById(R.id.pieChart)

        pieChart.setUsePercentValues(true)
        pieChart.description.isEnabled = false
        pieChart.setExtraOffsets(5f, 10f, 5f, 5f)

        // on below line we are setting drag for our pie chart
        pieChart.setDragDecelerationFrictionCoef(0.95f)

        // on below line we are setting hole
        // and hole color for pie chart
        pieChart.isDrawHoleEnabled = true
        pieChart.setHoleColor(Color.WHITE)

        // on below line we are setting circle color and alpha
        pieChart.setTransparentCircleColor(Color.WHITE)
        pieChart.setTransparentCircleAlpha(110)

        // on  below line we are setting hole radius
        pieChart.holeRadius = 58f
        pieChart.transparentCircleRadius = 61f

        // on below line we are setting center text
        pieChart.setDrawCenterText(true)

        // on below line we are setting
        // rotation for our pie chart
        pieChart.setRotationAngle(0f)

        // enable rotation of the pieChart by touch
        pieChart.isRotationEnabled = true
        pieChart.isHighlightPerTapEnabled = true

        // on below line we are setting animation for our pie chart
        pieChart.animateY(1400, Easing.EaseInOutQuad)

        // on below line we are setting the legend for pie chart
        val legend: Legend = pieChart.legend
        legend.isEnabled = false

        // pieChart.legend.isEnabled = false
        pieChart.setEntryLabelColor(Color.BLACK)
        pieChart.setEntryLabelTextSize(14f)

        // on below line we are creating array list and
        // adding data to it to display in pie chart
        val entries: ArrayList<PieEntry> = ArrayList()
        entries.add(PieEntry(carbsRatio, "Carbs")) //carbs
        entries.add(PieEntry(proteinRatio, "Protein")) // protein
        entries.add(PieEntry(fatRatio, "Fat")) // fat

        // on below line we are setting pie data set
        val dataSet = PieDataSet(entries, "Macros")

        // on below line we are setting icons.
        dataSet.setDrawIcons(false)

        // on below line we are setting slice for pie
        dataSet.sliceSpace = 3f
        dataSet.iconsOffset = MPPointF(0f, 40f)
        dataSet.selectionShift = 5f


        // add a lot of colors to list
        val colors: ArrayList<Int> = ArrayList()
        colors.add(ContextCompat.getColor(this, R.color.light_green))
        colors.add(ContextCompat.getColor(this, R.color.light_blue))
        colors.add(ContextCompat.getColor(this, R.color.light_yellow))

        // on below line we are setting colors.
        dataSet.colors = colors

        // on below line we are setting pie data set
        val data = PieData(dataSet)
        data.setValueFormatter(CustomValueFormatter())
        data.setValueTextSize(18f)
        data.setValueTypeface(Typeface.DEFAULT_BOLD)
        data.setValueTextColor(Color.BLACK)
        pieChart.setData(data)

        // undo all highlights
        pieChart.highlightValues(null)


        // loading chart
        pieChart.invalidate()
    }
    private fun setTextViewText(textViewId: Int, text: String) {
        val textView = findViewById<TextView>(textViewId)
        textView.text = text
    }
    class CustomValueFormatter : ValueFormatter() {
        //This is to put a percent symbol in the value label in the pie chart
        override fun getFormattedValue(value: Float): String {
            return "${value.toInt()}%" // Append percent symbol to the value
        }
    }
    private fun String.capitalizeWords(): String {
        return split(" ").joinToString(" ") { it ->
            it.replaceFirstChar {
            if (it.isLowerCase()) it.titlecase(
                Locale.ROOT
            ) else it.toString()
        } }
    }
}