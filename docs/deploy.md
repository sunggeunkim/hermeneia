# Deploy

Push to `main` builds `ghcr.io/sunggeunkim/hermeneia`, pushes the image, joins the tailnet, and on the guest runs `docker compose pull && up -d`. TLS is the existing Cloudflare tunnel `regamoment`, hostname `hermeneia.stugen.net`.

The Proxmox host is `192.168.68.220` (hostname `ai`). The RTX 3080 Ti is bound to vfio-pci. This guest is a new Ubuntu VM with PCI passthrough of `0000:01:00`. Do not clone CT 999: that container expects `/dev/nvidia0` on the host, and the host has no NVIDIA driver. Do not start VM 221 (`llama3.2-8b`) or VM 223 (`w11`) while this VM is up. They use the same PCI device. Do not install a driver on the Proxmox host, and do not edit CT 227 or CT 999.

Weights stay in a Docker volume (`/models`). The image does not contain them. `/health` is 200 as soon as the process listens, with `"ready": false` until the startup model has loaded.

## Guest, once

1. Create an Ubuntu 24.04 VM on `data-lvm-thinpool`: about 8 cores, 32 GB RAM, 128 GB disk. Pass through `0000:01:00` only. Give it a static LAN address.
2. On the guest: NVIDIA driver, Docker, nvidia-container-toolkit, Tailscale. MagicDNS name `hermeneia`. User `deploy`. App directory `/opt/hermeneia`. Install the public half of a new key `hermeneia_deploy_ci` for `deploy`.
3. Create `/opt/hermeneia/.env` (not in git):

```
HERMENEIA_TOKEN=...
HERMENEIA_MODEL=whisper
WHISPER_MODEL=large-v3
WHISPER_COMPUTE_TYPE=float16
HERMENEIA_DEVICE=cuda
```

The other checkpoint is downloaded the first time a request names it.

4. GitHub repo secrets on `sunggeunkim/hermeneia`, same names studyguide uses: `TS_OAUTH_CLIENT_ID`, `TS_OAUTH_SECRET`, `LXC_SSH_PRIVATE_KEY` (this guest's key, not studyguide's), `LXC_HOST` (`hermeneia`), `LXC_USER` (`deploy`).
5. In the Cloudflare Zero Trust dashboard, on tunnel `regamoment`, add `hermeneia.stugen.net` → `http://<guest-lan-ip>:8080`. Confirm the connector is running first (its documented LXC at `192.168.68.230` did not answer ping from the Proxmox host on 2026-09-27). Bypass cache for that hostname. There is no local tunnel config file.

## After the first deploy

```bash
curl -fsS https://hermeneia.stugen.net/health
```

`ready` should be true and `model` should be `whisper`. Then post one short WAV with the bearer token, and the same WAV again with `model=seamless-m4t-v2`. The second response's `load_ms` is the switch. A third call to that same model should report `load_ms` of 0. `nvidia-smi` inside the guest between those two resident states should show one checkpoint, not the sum.

Rollback on the guest:

```bash
cd /opt/hermeneia
IMAGE_TAG=sha-<7> docker compose -f docker-compose.prod.yml up -d
```

A manual pull needs your own `docker login ghcr.io`. The workflow's token is not left on the guest.
