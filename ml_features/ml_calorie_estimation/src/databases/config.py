from pydantic import BaseModel
from typing import Optional

class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    username: str
    password: str # will be set with environment variable
    host: str
    database: str
    port: Optional[int] = None # Added for RDS
    env: str = "local"
    
    @property
    def connection_string(self) -> str:
        if self.env == "production":
            conn_str = f'postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode=require'
        else:
            conn_str = f'postgresql://{self.username}:{self.password}@{self.host}/{self.database}'
        return conn_str
    