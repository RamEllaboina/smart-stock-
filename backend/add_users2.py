import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.database import SessionLocal, Base, engine
from backend.models.user import User, RoleEnum
from backend.core.security import get_password_hash

db = SessionLocal()

u1 = db.query(User).filter(User.email == "one@gmail.com").first()
if not u1:
    u1 = User(email="one@gmail.com", hashed_password=get_password_hash("one"), role=RoleEnum.MANAGER, tenant_id="t1", authorized_stores="Germany")
    db.add(u1)
else:
    u1.hashed_password = get_password_hash("one")

u2 = db.query(User).filter(User.email == "two@gmail.com").first()
if not u2:
    u2 = User(email="two@gmail.com", hashed_password=get_password_hash("two"), role=RoleEnum.MANAGER, tenant_id="t1", authorized_stores="France")
    db.add(u2)
else:
    u2.hashed_password = get_password_hash("two")

# Just in case the user types without .com
u3 = db.query(User).filter(User.email == "one@gmail").first()
if not u3:
    u3 = User(email="one@gmail", hashed_password=get_password_hash("One"), role=RoleEnum.MANAGER, tenant_id="t1", authorized_stores="Germany")
    db.add(u3)
else:
    u3.hashed_password = get_password_hash("One")

db.commit()
db.close()
print("Created successfully!")
