import time
import requests

class CoinGeckoClient:
    def __init__(self, api_key: str, rate_limit_per_min=250):
        self.api_key = api_key
        self.requests_per_min = 0
        self.rate_limit_per_min = rate_limit_per_min
        self.last_reset = time.time()

    def _throttle(self):
        now = time.time()
        if now - self.last_reset >= 60:
            self.requests_per_min = 0
            self.last_reset = now

        if self.requests_per_min >= self.rate_limit_per_min:
            sleep_time = 60 - (now - self.last_reset)
            print(f"Rate limit reached. Sleeping {sleep_time:.1f} seconds...")
            time.sleep(max(0, sleep_time))
            self.requests_per_min = 0
            self.last_reset = time.time()

    def get(self, url, params=None):
        self._throttle()
        headers = {"x-cg-pro-api-key": self.api_key}
        r = requests.get(url, params=params, headers=headers)

        self.requests_per_min += 1

        if r.status_code != 200:
            print("Error:", r.status_code, r.text)
            time.sleep(1)
            return self.get(url, params)

        return r.json()
