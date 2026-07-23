import uuid
import random
from locust import HttpUser, task, between

# Public Keys matching the mock behaviors in stellar.py
VALID_FUNDED_KEY = "GD3WGAA457DN2D6VDIRTSOPX2LQMRDLPIXPESFK7MY5LZ6WRBRDFH6LA"  # Ends with A -> 200 OK with balance
VALID_UNFUNDED_KEY = "GAGVSEDNTGJ2ITCKELISRDTKLEDQWCAUE2Q4SCU7QKX7M5FE3ZQTEWXB"  # Ends with B -> 404 Not Found
VALID_NO_TRUSTLINE_KEY = "GDFZPBIOURMAQBWPRXPKD27XAHLHPQRDGSSRY42SB4OC5G7ZBG6ENTUC"  # Ends with C -> 200 OK with no trustline
INVALID_KEY = "INVALID_STELLAR_KEY_123"  # -> 400 Bad Request

class UserBehavior(HttpUser):
    # Think time between tasks simulating human interaction
    wait_time = between(1, 3)

    def on_start(self):
        self.chat_id = None

    # Scenarios for /chat
    @task(3)
    def single_turn_chat(self):
        self.client.post("/chat", json={
            "prompt": "What is the importance of Fajr prayer?"
        }, name="/chat (single-turn)")

    @task(2)
    def multi_turn_chat_step(self):
        if not self.chat_id:
            self.chat_id = str(uuid.uuid4())
            self.client.post("/chat", json={
                "prompt": "Hello, explain Zakat?",
                "chat_id": self.chat_id
            }, name="/chat (multi-turn start)")
        else:
            self.client.post("/chat", json={
                "prompt": "How is it calculated on gold?",
                "chat_id": self.chat_id
            }, name="/chat (multi-turn follow-up)")
            
            # 50% chance to delete/end the conversation session
            if random.random() < 0.5:
                self.client.delete(f"/chat/{self.chat_id}", name="/chat (delete session)")
                self.chat_id = None

    # Scenarios for /zakat (stellar)
    @task(2)
    def zakat_valid_funded(self):
        self.client.post("/zakat", json={
            "public_key": VALID_FUNDED_KEY
        }, name="/zakat (valid funded)")

    @task(1)
    def zakat_valid_no_trustline(self):
        self.client.post("/zakat", json={
            "public_key": VALID_NO_TRUSTLINE_KEY
        }, name="/zakat (no trustline)")

    @task(1)
    def zakat_invalid_key(self):
        with self.client.post("/zakat", json={
            "public_key": INVALID_KEY
        }, catch_response=True, name="/zakat (invalid key 400)") as response:
            if response.status_code == 400:
                response.success()

    @task(1)
    def zakat_unfunded_key(self):
        with self.client.post("/zakat", json={
            "public_key": VALID_UNFUNDED_KEY
        }, catch_response=True, name="/zakat (unfunded 404)") as response:
            if response.status_code == 404:
                response.success()

    # Health Check Scenario
    @task(1)
    def health_check(self):
        self.client.get("/ping", name="/ping")
