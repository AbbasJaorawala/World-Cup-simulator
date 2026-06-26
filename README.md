# World Cup 2026 Simulator

Interactive Streamlit dashboard for simulating the 2026 FIFA World Cup.

## Run locally

```bash
pip install -r requirements.txt
streamlit run worldcup_simulator/dashboard/app.py
```

## Publish on Streamlit Community Cloud

1. Push this project to GitHub.
2. Create a new app at Streamlit Community Cloud.
3. Set the main file path to:

```text
worldcup_simulator/dashboard/app.py
```

4. Optional: add API keys in the Streamlit secrets panel if you want live API mode.

```toml
FOOTBALL_DATA_API_KEY = "..."
RAPIDAPI_KEY = "..."
SPORTAPI_RAPIDAPI_HOST = "sportapi7.p.rapidapi.com"
```

The dashboard also works with the included cached ratings and built-in group draw.
