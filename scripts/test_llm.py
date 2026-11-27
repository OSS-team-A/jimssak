"""Send a short chat request to the endpoint configured in backend/.env."""

from pathlib import Path

import httpx
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SmokeTestSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / "backend" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="LLM_",
    )

    base_url: str = ""
    api_key: SecretStr = SecretStr("")
    model: str = ""


def main() -> None:
    settings = SmokeTestSettings()
    api_key = settings.api_key.get_secret_value().strip()
    if not settings.base_url.strip() or not api_key:
        raise SystemExit("Set LLM_BASE_URL and LLM_API_KEY in backend/.env.")

    model = settings.model.strip()
    endpoint = "chat/completions" if model else "models"
    try:
        with httpx.Client(
            base_url=settings.base_url.strip(),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        ) as client:
            if not model:
                response = client.get(endpoint)
                response.raise_for_status()
                models = [entry["id"] for entry in response.json()["data"]]
                if len(models) != 1:
                    candidates = ", ".join(models) or "none"
                    raise SystemExit(
                        f"Set LLM_MODEL in backend/.env. Available models: {candidates}"
                    )
                model = models[0]

            endpoint = "chat/completions"
            response = client.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Reply in one short sentence."},
                        {"role": "user", "content": "Say hello from Jimssak."},
                    ],
                },
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as exc:
        hint = " Set LLM_MODEL to the provider's model ID." if endpoint == "models" else ""
        raise SystemExit(f"{endpoint} returned HTTP {exc.response.status_code}.{hint}") from None
    except httpx.RequestError as exc:
        raise SystemExit(f"Could not reach {endpoint}: {type(exc).__name__}.") from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise SystemExit(f"Unexpected response from {endpoint}.") from None

    if not isinstance(reply, str) or not reply.strip():
        raise SystemExit("The model returned an empty reply.")
    print(f"Model: {model}")
    print(f"Reply: {reply}")


if __name__ == "__main__":
    main()
