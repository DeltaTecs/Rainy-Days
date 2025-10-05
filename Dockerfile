FROM node:18-alpine

WORKDIR /usr/src/app

COPY app/package*.json ./

RUN npm install --production

COPY app/. ./

ENV NODE_ENV=production
ENV PORT=8080
EXPOSE 8080

CMD ["npm", "start"]
