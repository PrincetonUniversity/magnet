mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"minjie@ieee.org\"\n\
" > ~/.streamlit/credentials.toml

echo "[theme]
primaryColor = ‘#ff8f00’
secondaryBackgroundColor = ‘#c3cfe0’
textColor= ‘#000000’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml