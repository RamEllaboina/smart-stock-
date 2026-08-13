from sqlalchemy import Column, Integer, String, Boolean, Enum
import enum
from backend.core.database import Base

class RoleEnum(str, enum.Enum):
    ADMIN = "ADMIN"
    MANAGER = "MANAGER"
    STAFF = "STAFF"
    READ_ONLY = "READ_ONLY"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.STAFF, nullable=False)
    tenant_id = Column(String, nullable=False)
    # comma separated store ids for authorization check
    authorized_stores = Column(String, nullable=True) 
    is_active = Column(Boolean, default=True)
