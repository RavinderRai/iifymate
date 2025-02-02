from pydantic import BaseModel

class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    username: str
    password: str
    host: str = "localhost"
    database: str = "recipe_data"
    
    @property
    def connection_string(self) -> str:
        return f'postgresql://{self.username}:{self.password}@{self.host}/{self.database}'