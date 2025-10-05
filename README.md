# Rainy-Days

A containerized web stack consisting of:

- **nginx (reverse proxy + static file server + TLS termination)**
- **web** (Node.js frontend, serves SPA build on port 8080 inside container)
- **api** (Python / Flask or similar on port 5000)

The stack is orchestrated with `docker compose` and supports HTTPS via Let's Encrypt certificates whose paths are injected through environment variables and expanded into an NGINX template at container start.


---
## Environment Variables (.env)
The project expects a `.env` file in the repository root. Example:
```dotenv
# Path on the HOST where Let's Encrypt stores certs (mounted read-only into nginx container)
CERTBOT_PATH=/etc/letsencrypt

# The fullchain + private key that nginx should use (inside container these resolve via envsubst)
# For production these should point at real issued certs for your domain.
SSL_CERT_PATH=/etc/letsencrypt/live/example.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/example.com/privkey.pem
```
Notes:
- The paths above are the *container* paths after mounting. `docker-compose.yml` mounts `${CERTBOT_PATH}` from the host to the same path inside the container.
- If the files do not exist when nginx starts, the container will fail with an SSL error.

---
## TLS / HTTPS with Let's Encrypt (Production)
To enable HTTPS you must first obtain valid certificates for your domain. There are multiple strategies; below is a recommended host-level issuance using the standard Certbot client.

### 1. Prerequisites
- A domain name (e.g., `example.com`).
- DNS A (and/or AAAA) records pointing to the server's public IP.
- Ports 80 and 443 reachable from the internet (no intervening firewall blocks).

### 2. Stop nginx (if already running and you want to use standalone mode)
If you plan to use the `--standalone` method Certbot must bind port 80:
```bash
docker compose stop nginx
```

### 3. Install Certbot (host)
(Example for Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install -y certbot
```

### 4. Creating a certificate and running the setup
```bash
sudo certbot certonly --standalone -d example.com -d www.example.com --agree-tos -m you@example.com --no-eff-email
```
This places certs under `/etc/letsencrypt/live/example.com/`.

Then update `.env`:
```dotenv
SSL_CERT_PATH=/etc/letsencrypt/live/example.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/example.com/privkey.pem
```
Restart nginx:
```bash
docker compose up -d nginx
```
