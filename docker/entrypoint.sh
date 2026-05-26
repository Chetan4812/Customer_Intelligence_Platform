#!/bin/sh
# Add Azure App Service internal hostname to /etc/hosts so the container can resolve its own public URL
if [ -n "$WEBSITE_HOSTNAME" ]; then
  echo "127.0.0.1 $WEBSITE_HOSTNAME" >> /etc/hosts
  echo "DEBUG: WEBSITE_HOSTNAME=$WEBSITE_HOSTNAME" >> /tmp/entrypoint_debug.log
fi
# Start Streamlit on port 8501 (matching WEBSITES_PORT in Azure)
exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0

