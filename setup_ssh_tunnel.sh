#!/bin/bash

# SSH Tunnel Configuration
SSH_HOST="13.232.56.71"
SSH_USER="sshtunnel"
SSH_PASSWORD="Qwerty@123"
SSH_PORT="22"

# Database Configuration
DB_HOST="crm-kulish-dev.cqxcmnpnysxs.ap-south-1.rds.amazonaws.com"
DB_PORT="5432"
LOCAL_PORT="5433"  # Local port to forward to

echo "🚀 Setting up SSH tunnel to PostgreSQL database..."
echo "=================================================="
echo "SSH Host: $SSH_HOST"
echo "SSH User: $SSH_USER"
echo "Database Host: $DB_HOST"
echo "Local Port: $LOCAL_PORT"
echo ""

# Kill any existing tunnel on the local port
echo "🔍 Checking for existing tunnels on port $LOCAL_PORT..."
lsof -ti:$LOCAL_PORT | xargs kill -9 2>/dev/null || echo "No existing tunnels found."

# Create SSH tunnel
echo "🔗 Creating SSH tunnel..."
echo "Local port $LOCAL_PORT -> $SSH_HOST -> $DB_HOST:$DB_PORT"
echo ""
echo "Press Ctrl+C to stop the tunnel when done testing."
echo ""

# Create the tunnel
ssh -L $LOCAL_PORT:$DB_HOST:$DB_PORT $SSH_USER@$SSH_HOST -N -v 