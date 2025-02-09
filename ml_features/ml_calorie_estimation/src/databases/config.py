from pydantic import BaseModel

class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    username: str
    password: str # will be set with environment variable
    host: str
    database: str
    
    @property
    def connection_string(self) -> str:
        return f'postgresql://{self.username}:{self.password}@{self.host}/{self.database}'