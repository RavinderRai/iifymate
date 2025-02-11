INGREDIENT_LIST_PROMPT_TEMPLATE = """
Analyze this food image and provide all visible ingredients in a structured JSON format.

Instructions:
- Include precise quantities and units when visible (e.g., "2 tablespoons" not "2 tbsp")
- Add preparation methods if visible (e.g., "diced", "thinly sliced")
- For ingredients without visible quantities, list only the ingredient name
- Use standard cooking units: tablespoon, teaspoon, cup, ounce, pound
- List ALL visible ingredients, no matter how small
- Keep formatting consistent throughout the list

Example input: [Image of a Moroccan Eggplant dish]
Example output:
{
  "ingredients": [
    "1 tablespoon olive oil",
    "1 large eggplant, cut into 1-inch pieces",
    "1 large onion, thinly sliced",
    "2 medium carrots, cut into 1/2 inch pieces",
    "1 can whole tomatoes in juice",
    "2 cloves garlic, minced",
    "1 tablespoon ras el hanout",
    "1 teaspoon cumin",
    "1/4 teaspoon cayenne pepper",
    "salt",
    "pepper",
    "cilantro"
  ]
}

Now analyze the provided image and output the ingredients in the same JSON format.
"""
