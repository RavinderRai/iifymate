from typing import Final

# final to raise an error if the list is modified
DIET_LABELS: Final[list[str]] = [
    "balanced",      # Protein/Fat/Carb values in 15/35/50 ratio
    "high-fiber",    # More than 5g fiber per serving
    "high-protein",  # More than 50% of total calories from proteins
    "low-carb",      # Less than 20% of total calories from carbs
    "low-fat",       # Less than 15% of total calories from fat
    "low-sodium"     # Less than 140mg Na per serving
]

HEALTH_LABELS: Final[list[str]] = [
    "alcohol-cocktail",    # Describes an alcoholic cocktail
    "alcohol-free",        # No alcohol used or contained
    "celery-free",        # Does not contain celery or derivatives
    "crustacean-free",    # Does not contain crustaceans
    "dairy-free",         # No dairy; no lactose
    "DASH",               # Dietary Approaches to Stop Hypertension diet
    "egg-free",           # No eggs or products containing eggs
    "fish-free",          # No fish or fish derivatives
    "fodmap-free",        # Does not contain FODMAP foods
    "gluten-free",        # No ingredients containing gluten
    "immuno-supportive",  # Science-based immune system strengthening
    "keto-friendly",      # Maximum 7 grams of net carbs per serving
    "kidney-friendly",    # Restricted phosphorus, potassium, and sodium
    "kosher",             # Contains only kosher-allowed ingredients
    "low-potassium",      # Less than 150mg per serving
    "low-sugar",          # No simple sugars
    "lupine-free",        # Does not contain lupine or derivatives
    "Mediterranean",      # Mediterranean diet
    "mollusk-free",       # No mollusks
    "mustard-free",       # Does not contain mustard or derivatives
    "No-oil-added",       # No oil added except in basic ingredients
    "paleo",              # Excludes agricultural products
    "peanut-free",        # No peanuts or products containing peanuts
    "pecatarian",         # No meat, can contain dairy and fish
    "pork-free",          # Does not contain pork or derivatives
    "red-meat-free",      # No red meat or products containing red meat
    "sesame-free",        # Does not contain sesame seed or derivatives
    "shellfish-free",     # No shellfish or shellfish derivatives
    "soy-free",           # No soy or products containing soy
    "sugar-conscious",    # Less than 4g of sugar per serving
    "sulfite-free",       # No Sulfites
    "tree-nut-free",      # No tree nuts or products containing tree nuts
    "vegan",              # No animal products
    "vegetarian",         # No meat, poultry, or fish
    "wheat-free"          # No wheat, can have gluten though
]

MEAL_TYPES: Final[list[str]] = [
    "breakfast",
    "brunch",
    "lunch", # lunch/dinner are the same according to the API, so we only need one of these labels
    "snack",
    "teatime"
]

DISH_TYPES: Final[list[str]] = [
    "Alcohol Cocktail",
    "Biscuits and Cookies",
    "Bread",
    "Cereals",
    "Condiments and Sauces",
    "Desserts",
    "Drinks",
    "Egg",
    "Ice Cream and Custard",
    "Main Course",
    "Pancake",
    "Pasta",
    "Pastry",
    "Pies and Tarts",
    "Pizza",
    "Preps",
    "Preserve",
    "Salad",
    "Sandwiches",
    "Seafood",
    "Side Dish",
    "Soup",
    "Special Occasions",
    "Starter",
    "Sweets"
]

CUISINE_TYPES: Final[list[str]] = [
    "American",
    "Asian",
    "British",
    "Caribbean",
    "Central Europe",
    "Chinese",
    "Eastern Europe",
    "French",
    "Greek",
    "Indian",
    "Italian",
    "Japanese",
    "Korean",
    "Kosher",
    "Mediterranean",
    "Mexican",
    "Middle Eastern",
    "Nordic",
    "South American",
    "South East Asian",
    "World"  # International cuisine/Other
]