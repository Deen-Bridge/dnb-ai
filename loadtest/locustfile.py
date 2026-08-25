import os
import uuid

from locust import HttpUser, between, task

VALID_KEY = os.getenv("LOADTEST_VALID_STELLAR_KEY", "GBBD47IF6LWK7P7MDEVSCWR7DPUWV3NY3DTQEVFL4NAT4AQH3ZLLFLA5")
UNKNOWN_KEY = os.getenv("LOADTEST_UNKNOWN_STELLAR_KEY", "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")


class ChatLoadUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        self.chat_id = str(uuid.uuid4())

    @task(3)
    def chat_single_turn(self):
        self.client.post("/chat", json={"prompt": "What is sadaqah?"}, name="chat-single")

    @task(2)
    def chat_multi_turn(self):
        self.client.post("/chat", json={"prompt": "Explain zakat.", "chat_id": self.chat_id}, name="chat-multi-1")
        self.client.post(
            "/chat", json={"prompt": "Give me a short example.", "chat_id": self.chat_id}, name="chat-multi-2"
        )

    @task(1)
    def zakat_success(self):
        self.client.post("/zakat", json={"public_key": VALID_KEY}, name="zakat-success")

    @task(1)
    def zakat_invalid_key(self):
        with self.client.post(
            "/zakat",
            json={"public_key": "not-a-key"},
            name="zakat-400",
            catch_response=True,
        ) as response:
            if response.status_code == 400:
                response.success()
            else:
                response.failure(f"expected 400, got {response.status_code}")

    @task(1)
    def zakat_unknown_account(self):
        with self.client.post(
            "/zakat",
            json={"public_key": UNKNOWN_KEY},
            name="zakat-404",
            catch_response=True,
        ) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure(f"expected 404, got {response.status_code}")
