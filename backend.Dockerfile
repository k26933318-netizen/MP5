FROM node:20-slim

WORKDIR /app
COPY package.json package-lock.json ./
COPY backend/package.json ./backend/package.json
COPY blockchain/chaincode/voting-chaincode/package.json ./blockchain/chaincode/voting-chaincode/package.json
RUN npm ci --omit=dev
COPY backend ./backend
COPY public ./public
CMD ["npm", "start"]
