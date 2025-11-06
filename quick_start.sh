#!/bin/bash

# Quick Start Script for News Fact Validation Graph
# This script sets up the environment and runs the demo

echo "=========================================="
echo "News Fact Validation Graph - Quick Start"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Please edit .env and add your OpenAI API key!"
    echo ""
    echo "   1. Open .env file"
    echo "   2. Replace 'your_openai_api_key_here' with your actual API key"
    echo "   3. Save the file"
    echo "   4. Run: python demo.py"
    echo ""
    echo "   Get your API key from: https://platform.openai.com/api-keys"
    echo ""
else
    echo "✓ .env file exists"
    echo ""

    # Check if API key is set
    if grep -q "your_openai_api_key_here" .env; then
        echo "⚠️  WARNING: OpenAI API key not set in .env file!"
        echo "   Please edit .env and add your API key before running the demo."
        echo ""
    else
        echo "✓ OpenAI API key is configured"
        echo ""

        # Create output directories
        mkdir -p outputs/graphs

        # Run the demo
        echo "🚀 Running demo..."
        echo ""
        python demo.py
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  • Edit .env file with your OpenAI API key (if not done)"
echo "  • Run: python demo.py"
echo "  • View outputs in: outputs/graphs/"
echo ""
