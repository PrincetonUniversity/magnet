mkdir -p ~/.streamlit/
echo "[theme]\n\
primaryColor = ‘#84a3a7’\n\
backgroundColor = ‘#EFEDE8’\n\
secondaryBackgroundColor = ‘#fafafa’\n\
textColor= ‘#424242’\n\
font = ‘sans serif’\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml