HEALTH_LABEL_PROMPT_TEMPLATE = """
Analyze these ingredients and determine the most appropriate single health label in JSON format.

Ingredients: {{ ingredients | tojson(indent=2) }}

Instructions:
- Choose exactly one label from: ['Vegan', 'Vegetarian', 'Pescatarian', 'Paleo', 'Red-Meat-Free', 'Mediterranean']
- Consider these rules:
    * Vegan: No animal products whatsoever
    * Vegetarian: May include dairy/eggs but no meat/fish
    * Pescatarian: Includes fish but no other meat
    * Paleo: No grains, dairy, processed foods
    * Red-Meat-Free: May include poultry/fish
    * Mediterranean: Emphasizes plant-based foods, fish, olive oil

Example input: [
        "1 tablespoon olive oil",
        "1 large eggplant, cut into 1-inch pieces",
        "1 large brown or yellow onion, thinly slices",
        "2 medium carrots cut into 1/2 inch pieces",
        "1 can of whole tomatoes in juices",
        "2 cloves of garlic, minced or finely chopped",
        "1 tablespoon ras el hanout",
        "1 teaspoon cumin",
        "1/4 teaspoon hot chili pepper/cayenne pepper",
        "salt",
        "pepper",
        "cilantro"
]
Example output:
{
    "health_label": "Vegan"
}

Return the label in the same JSON format.
"""

RECIPE_LABEL_PROMPT_TEMPLATE = """
Create an appropriate recipe name for a dish with these ingredients in JSON format.

Ingredients: {{ ingredients | tojson(indent=2) }}

Instructions:
- Create a descriptive but concise name (3-6 words)
- Include main ingredients or cooking method
- Use title case
- Consider key flavors and spices (e.g., Moroccan spices)
- Don't include words like "recipe" or "dish"

Example input: [
    "1 tablespoon olive oil",
    "1 large eggplant, cut into 1-inch pieces",
    "1 large brown or yellow onion, thinly slices",
    "2 medium carrots cut into 1/2 inch pieces",
    "1 can of whole tomatoes in juices",
    "2 cloves of garlic, minced or finely chopped",
    "1 tablespoon ras el hanout",
    "1 teaspoon cumin",
    "1/4 teaspoon hot chili pepper/cayenne pepper",
    "salt",
    "pepper",
    "cilantro"
]
Example output:
{
    "recipe_label": "Slow Cooker Moroccan Eggplant"
}

Return the name in the same JSON format.
"""