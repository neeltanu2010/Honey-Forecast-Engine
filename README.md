# Hive Forecast Engine — Financify SaaS Tool

A premium Streamlit tool that forecasts next-year revenue, net income, EPS and implied stock price range using bear/base/bull cases.

## Features

- Financify yellow/black money-bees theme
- Light honeycomb background
- Mobile and desktop responsive layout
- Login + subscription gate hooks
- SureCart upgrade button
- Yahoo Finance data fetch
- XGBoost if available, with fallback models
- Bear/Base/Bull scenario forecast
- Historical PE benchmarking
- Forecast CSV download

## Install

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Secrets

Add these in `.streamlit/secrets.toml`:

```toml
BACKEND_URL = "https://your-financify-backend.onrender.com"
SURECART_CHECKOUT_URL = "https://financify.blog/buy/financify-tools"
DEV_MODE = "false"
```

For local testing without login/subscription, use:

```toml
DEV_MODE = "true"
```

## Expected Backend Endpoints

The app expects the same SaaS-style backend pattern as your previous Financify tools:

### Send login code

`POST /auth/send-code`

```json
{"email":"user@example.com"}
```

### Verify login code

`POST /auth/verify-code`

```json
{"email":"user@example.com", "code":"123456"}
```

Expected response:

```json
{"token":"jwt_or_session_token"}
```

### Check subscription

`POST /subscription/status`

```json
{"email":"user@example.com", "token":"jwt_or_session_token"}
```

Expected response:

```json
{"active":true, "plan":"premium"}
```

## Forecast Logic

The app does not predict one fixed stock price. It forecasts next-year fundamentals first:

- Revenue growth
- Net margin
- EPS growth

Then it values the company using:

```text
Forecast EPS × Applied PE = Implied Stock Price
```

Bear/Base/Bull cases use different growth, margin and PE assumptions.

## Important Note

Yahoo Finance may not provide full 20-year financial statement history for every stock. The app uses the available annual history up to 20 years.

This tool is for education and research only. It is not financial advice.
