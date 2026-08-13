import os
import json
from twilio.rest import Client
from dotenv import load_dotenv

# Load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

def send_whatsapp_alert(product_id: str, current_stock: int, forecast_demand: int, 
                       safety_stock: int, recommended_reorder: int, force_mock: bool = False):
                       
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_whatsapp_number = os.getenv('TWILIO_WHATSAPP_FROM') # e.g. 'whatsapp:+14155238886'
    to_whatsapp_number = os.getenv('WHATSAPP_TO')
    
    message_body = (
        f"🚨 *Smart Stock Alert*\n\n"
        f"Product: {product_id}\n"
        f"Current Stock: {current_stock}\n"
        f"Forecast Demand: {forecast_demand}\n"
        f"Safety Stock: {safety_stock}\n"
        f"Recommended Reorder: {recommended_reorder}\n\n"
        f"Smart Stock recommends restocking this product."
    )
    
    if force_mock or not account_sid or not auth_token:
        print(f"[MOCK WHATSAPP ALERT SENT]")
        print(message_body)
        return {"status": "mock_sent", "message": message_body}
        
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message_body,
            from_=from_whatsapp_number,
            to=to_whatsapp_number
        )
        print(f"[WHATSAPP ALERT SENT] SID: {message.sid}")
        return {"status": "sent", "sid": message.sid, "message": message_body}
    except Exception as e:
        print(f"Failed to send real WhatsApp alert. Using fallback mock. Error: {e}")
        print(f"[MOCK WHATSAPP ALERT] {message_body}")
        return {"status": "error_fallback_mock", "error": str(e), "message": message_body}
        