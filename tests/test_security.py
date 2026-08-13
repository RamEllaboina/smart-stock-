import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.database import SessionLocal
from backend.models.user import User, RoleEnum
from backend.core.security import get_password_hash

client = TestClient(app)

@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    # Create test users
    if not db.query(User).filter(User.email == "test_admin@smartstock.local").first():
        db.add(User(
            email="test_admin@smartstock.local",
            hashed_password=get_password_hash("password123"),
            role=RoleEnum.ADMIN,
            tenant_id="test_tenant",
            authorized_stores="*"
        ))
        db.add(User(
            email="test_staff@smartstock.local",
            hashed_password=get_password_hash("password123"),
            role=RoleEnum.STAFF,
            tenant_id="test_tenant",
            authorized_stores="store_01"
        ))
        db.commit()
    yield db
    # Cleanup could happen here
    db.close()

def test_login_success(db_session):
    response = client.post(
        "/auth/login",
        data={"username": "test_admin@smartstock.local", "password": "password123"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["user"]["role"] == "ADMIN"

def test_login_failure():
    response = client.post(
        "/auth/login",
        data={"username": "wrong@user", "password": "bad"}
    )
    assert response.status_code == 401

def test_admin_access(db_session):
    login = client.post("/auth/login", data={"username": "test_admin@smartstock.local", "password": "password123"})
    token = login.json()["access_token"]
    
    # We trigger a mock retraining which needs ADMIN
    response = client.post(
        "/monitoring/retrain",
        json={"store_id": "store_01", "product_id": "P001"},
        headers={"Authorization": f"Bearer {token}"}
    )
    # The route returns 404 since there is no report generated yet, but NOT 403 Authorization!
    assert response.status_code in [404, 200, 400]
    assert response.status_code != 403

def test_staff_unauthorized_admin(db_session):
    login = client.post("/auth/login", data={"username": "test_staff@smartstock.local", "password": "password123"})
    token = login.json()["access_token"]
    
    # Staff trying to trigger retraining
    response = client.post(
        "/monitoring/retrain",
        json={"store_id": "store_01", "product_id": "P001"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert "Not enough permissions" in response.json()['detail']

def test_store_level_authorization(db_session):
    login = client.post("/auth/login", data={"username": "test_staff@smartstock.local", "password": "password123"})
    token = login.json()["access_token"]
    
    # Staff authorized for store_01
    response = client.post(
        "/inventory/recommendations",
        json={"store_id": "store_01", "product_id": "P001", "current_stock": 50},
        headers={"Authorization": f"Bearer {token}"}
    )
    # Could be 400/404 based on data existence, but must NOT be 403
    assert response.status_code != 403
    
    # Staff unauthorized for store_02
    response2 = client.post(
        "/inventory/recommendations",
        json={"store_id": "store_02", "product_id": "P001", "current_stock": 50},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response2.status_code == 403
    assert "Access denied to store" in response2.json()['detail']
