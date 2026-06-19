import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from aurum.config import (
    CTRADER_ACCESS_TOKEN,
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_REDIRECT_URI,
    CTRADER_REFRESH_TOKEN,
    CTRADER_TOKENS_PATH,
)

logger = logging.getLogger(__name__)
TOKEN_URL = "https://openapi.ctrader.com/apps/token"
AUTH_URL = (
    "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
    "?client_id={client_id}&redirect_uri={redirect_uri}&scope=trading&product=web"
)


@dataclass
class TokenBundle:
    access_token: str
    refresh_token: str
    expires_in: int = 2_628_000
    updated_at: str = ""

    def __post_init__(self):
        if not self.updated_at:
            self.updated_at = datetime.now(timezone.utc).isoformat()


class CTraderAuth:
    def __init__(
        self,
        client_id: str = CTRADER_CLIENT_ID,
        client_secret: str = CTRADER_CLIENT_SECRET,
        redirect_uri: str = CTRADER_REDIRECT_URI,
        tokens_path: Path = CTRADER_TOKENS_PATH,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.tokens_path = tokens_path

    def get_auth_url(self) -> str:
        return AUTH_URL.format(client_id=self.client_id, redirect_uri=self.redirect_uri)

    def exchange_code(self, code: str) -> TokenBundle:
        resp = requests.get(
            TOKEN_URL,
            params={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorCode"):
            raise RuntimeError(data.get("description") or data.get("errorCode"))
        bundle = TokenBundle(
            access_token=data["accessToken"],
            refresh_token=data["refreshToken"],
            expires_in=int(data.get("expiresIn", 2_628_000)),
        )
        self.save_tokens(bundle)
        return bundle

    def refresh(self, refresh_token: str | None = None) -> TokenBundle:
        token = refresh_token or self.load_tokens().refresh_token
        resp = requests.get(
            TOKEN_URL,
            params={
                "grant_type": "refresh_token",
                "refresh_token": token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorCode"):
            raise RuntimeError(data.get("description") or data.get("errorCode"))
        bundle = TokenBundle(
            access_token=data["accessToken"],
            refresh_token=data["refreshToken"],
            expires_in=int(data.get("expiresIn", 2_628_000)),
        )
        self.save_tokens(bundle)
        return bundle

    def save_tokens(self, bundle: TokenBundle) -> None:
        self.tokens_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokens_path.write_text(json.dumps(asdict(bundle), indent=2), encoding="utf-8")

    def load_tokens(self) -> TokenBundle:
        if self.tokens_path.exists():
            raw = json.loads(self.tokens_path.read_text(encoding="utf-8"))
            return TokenBundle(**raw)
        if CTRADER_ACCESS_TOKEN and CTRADER_REFRESH_TOKEN:
            bundle = TokenBundle(
                access_token=CTRADER_ACCESS_TOKEN,
                refresh_token=CTRADER_REFRESH_TOKEN,
            )
            self.save_tokens(bundle)
            return bundle
        raise FileNotFoundError(
            "cTrader tokens not found. Complete OAuth in UI or set CTRADER_ACCESS_TOKEN in .env"
        )

    def get_access_token(self) -> str:
        return self.load_tokens().access_token

    def ensure_valid_token(self) -> str:
        """Refresh access token if close to expiry."""
        bundle = self.load_tokens()
        try:
            updated = datetime.fromisoformat(bundle.updated_at.replace("Z", "+00:00"))
            age_sec = (datetime.now(timezone.utc) - updated).total_seconds()
            if age_sec > bundle.expires_in - 86400:
                bundle = self.refresh(bundle.refresh_token)
        except Exception:
            pass
        return bundle.access_token
