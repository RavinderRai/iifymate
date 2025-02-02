from sqlalchemy import Column, Integer, String, JSON, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class CleanedRecipe(Base):
    __tablename__ = 'cleaned_recipes'
    id = Column(Integer, primary_key=True)
    raw_recipe_id = Column(Integer, ForeignKey('raw_recipes.id'))
    # ... cleaned data columns
    raw_recipe = relationship("RawRecipe")