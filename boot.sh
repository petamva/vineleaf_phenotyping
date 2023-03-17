#!/bin/bash
source venv/bin/activate
exec gunicorn -b :5000 --timeout 200 --workers=2 --threads=4 --worker-class=gthread --access-logfile - --error-logfile - app.main:app