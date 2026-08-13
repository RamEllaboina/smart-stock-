from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum
import datetime
from backend.core.database import Base

class ListingStatus(str, enum.Enum):
    LISTED = "LISTED"
    MATCHED = "MATCHED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

class TransactionStatus(str, enum.Enum):
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    CONFIRMED = "CONFIRMED"
    IN_TRANSIT = "IN_TRANSIT"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class SurplusListing(Base):
    __tablename__ = "surplus_listings"
    
    id = Column(String, primary_key=True, index=True)
    seller_store_id = Column(String, index=True)
    product_id = Column(String, index=True)
    available_qty = Column(Integer)
    min_qty = Column(Integer, default=1)
    price_per_unit = Column(Float)
    expiry_date = Column(DateTime, nullable=True) # Important for perishable
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default=ListingStatus.LISTED.value)

class ExchangeTransaction(Base):
    __tablename__ = "exchange_transactions"
    
    id = Column(String, primary_key=True, index=True)
    listing_id = Column(String, ForeignKey("surplus_listings.id"))
    buyer_store_id = Column(String, index=True)
    requested_qty = Column(Integer)
    status = Column(String, default=TransactionStatus.REQUESTED.value)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    listing = relationship("SurplusListing")
