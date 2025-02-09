import ast
import json
import pandas as pd

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary columns."""
    columns_to_drop = ['uri', 'url', 'cautions', 'totalDaily', 'digest']
    return df.drop(columns=columns_to_drop)

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for consistency."""
    return df.rename(columns={'yield_': 'serving_size'})

def clean_ingredients(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the ingredients column."""
    def parse_ingredients(ingredients):
        if isinstance(ingredients, str):
            ingredients = json.loads(ingredients)
            # old method
            # ingredients = ast.literal_eval(ingredients)
        return [{'text': ing['text'], 
                'quantity': float(ing['quantity']) if 'quantity' in ing else 0.0,
                'measure': ing['measure'] if 'measure' in ing else 'unit'} 
                for ing in ingredients]
    
    df['ingredients'] = df['ingredients'].apply(parse_ingredients)
    return df

def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric type."""
    numeric_columns = ['calories', 'totalWeight', 'totalTime']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def clean_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the nutrients data."""
    def parse_nutrients(nutrients):
        if isinstance(nutrients, str):
            nutrients = ast.literal_eval(nutrients)
        return {v['label']: {'quantity': v['quantity'], 'unit': v['unit']} 
                for k, v in nutrients.items()}
    
    df['totalNutrients'] = df['totalNutrients'].apply(parse_nutrients)
    return df

def remove_faulty_nutrients(df: pd.DataFrame) -> pd.DataFrame:
    """Remove records with invalid nutrient data."""
    faulty_indices = []
    
    for index, nutrients in df.totalNutrients.items():
        for nutrient in ['FAT', 'CHOCDF.net', 'PROCNT']:
            if (nutrients.get(nutrient) is None or
                nutrients[nutrient].get('quantity') is None or
                nutrients[nutrient]['quantity'] < 0 or
                nutrients[nutrient].get('unit') is None or
                nutrients[nutrient]['unit'] != 'g'):
                faulty_indices.append(index)
                break
    
    return df.drop(index=list(set(faulty_indices)))
