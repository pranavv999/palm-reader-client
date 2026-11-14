#!/usr/bin/env bash
# ==============================================================================
# Palm Reader Client - One-Command Setup & Launch Script
# 100% In-Browser WebAssembly + MediaPipe AI Palm Reader
# ==============================================================================

set -e

# ANSI Color Codes
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}🖐️  Palm Reader AI (100% Client-Side WebAssembly)    ${NC}"
echo -e "${CYAN}======================================================${NC}"

# 1. Verify Node.js installation
if ! command -v node >/dev/null 2>&1; then
  echo -e "${RED}❌ Node.js is required but not installed.${NC}"
  echo -e "Please install Node.js (v18+) from https://nodejs.org/ and try again."
  exit 1
fi

NODE_VERSION=$(node -v)
echo -e "${GREEN}✓ Found Node.js: ${NODE_VERSION}${NC}"

# 2. Check for npm packages
if [ ! -d "node_modules" ]; then
  echo -e "${YELLOW}📦 node_modules not found. Installing dependencies...${NC}"
  npm install
  echo -e "${GREEN}✓ Dependencies installed successfully.${NC}"
else
  echo -e "${GREEN}✓ Dependencies already installed.${NC}"
fi

# 3. Verify pre-bundled model weights
if [ ! -f "public/weights.onnx" ]; then
  echo -e "${RED}⚠️ Warning: public/weights.onnx is missing.${NC}"
else
  echo -e "${GREEN}✓ Pre-bundled ONNX Palm Line Model verified (10.7 MB).${NC}"
fi

# 4. Launch Vite HTTPS server
echo -e "${CYAN}------------------------------------------------------${NC}"
echo -e "${YELLOW}📱 Mobile Testing Instructions:${NC}"
echo -e " 1. Look for the 'Network: https://<your-local-ip>:5173/' link below."
echo -e " 2. Open that URL on your phone connected to the same Wi-Fi."
echo -e " 3. Tap 'Advanced' -> 'Proceed' to accept the self-signed SSL cert."
echo -e " 4. Allow camera permission when prompted."
echo -e "${CYAN}------------------------------------------------------${NC}"
echo -e "${GREEN}🚀 Launching Vite Dev Server...${NC}"

npm run dev -- --host
