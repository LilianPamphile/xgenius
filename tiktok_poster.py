# tiktok_poster.py
# -*- coding: utf-8 -*-
import os, time, json, math, mimetypes
import requests

TIKTOK_API_BASE = os.getenv("TIKTOK_API_BASE", "https://open-api.tiktok.com")  # laisse par défaut

CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")
REDIRECT_URI = os.getenv("TIKTOK_REDIRECT_URI")

ACCESS_TOKEN = os.getenv("TIKTOK_ACCESS_TOKEN")          # à générer une 1ère fois via OAuth (voir doc)
REFRESH_TOKEN = os.getenv("TIKTOK_REFRESH_TOKEN")        # idem
ACCESS_TOKEN_EXPIRES_AT = float(os.getenv("TIKTOK_ACCESS_EXPIRES_AT", "0"))  # timestamp (epoch)

def _post(url, headers=None, json_body=None, data=None, files=None, timeout=30):
    h = {"User-Agent":"xgenius/1.0"}
    if headers: h.update(headers)
    r = requests.post(url, headers=h, json=json_body, data=data, files=files, timeout=timeout)
    try:
        out = r.json()
    except Exception:
        out = {"status_code": r.status_code, "text": r.text}
    if r.status_code >= 300:
        raise RuntimeError(f"HTTP {r.status_code} on {url} -> {out}")
    return out

def _refresh_access_token(refresh_token=None):
    """Échange REFRESH_TOKEN contre un nouvel access_token + expiry."""
    refresh_token = refresh_token or os.getenv("TIKTOK_REFRESH_TOKEN")
    if not (CLIENT_KEY and CLIENT_SECRET and refresh_token):
        raise RuntimeError("TikTok OAuth incomplet: CLIENT_KEY/CLIENT_SECRET/REFRESH_TOKEN requis.")
    url = f"{TIKTOK_API_BASE}/v2/oauth/token/"
    body = {
        "client_key": CLIENT_KEY,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    resp = _post(url, json_body=body)
    # structure type attendue
    access_token = resp.get("access_token") or resp.get("data", {}).get("access_token")
    refresh_token = resp.get("refresh_token") or resp.get("data", {}).get("refresh_token")
    expires_in = resp.get("expires_in") or resp.get("data", {}).get("expires_in", 3600)
    if not access_token:
        raise RuntimeError(f"Réponse inattendue refresh token: {resp}")
    # Persiste en variables d'env du process (utile pour ce run)
    os.environ["TIKTOK_ACCESS_TOKEN"] = access_token
    os.environ["TIKTOK_REFRESH_TOKEN"] = refresh_token or os.getenv("TIKTOK_REFRESH_TOKEN", "")
    os.environ["TIKTOK_ACCESS_EXPIRES_AT"] = str(time.time() + int(expires_in) - 60)
    return access_token

def ensure_access_token():
    """Retourne un access_token valide (refresh si besoin)."""
    global ACCESS_TOKEN, ACCESS_TOKEN_EXPIRES_AT
    ACCESS_TOKEN = os.getenv("TIKTOK_ACCESS_TOKEN")
    try:
        ACCESS_TOKEN_EXPIRES_AT = float(os.getenv("TIKTOK_ACCESS_EXPIRES_AT", "0"))
    except Exception:
        ACCESS_TOKEN_EXPIRES_AT = 0.0
    if ACCESS_TOKEN and time.time() < ACCESS_TOKEN_EXPIRES_AT:
        return ACCESS_TOKEN
    # sinon on tente le refresh
    return _refresh_access_token()

def _guess_mime(path):
    mt, _ = mimetypes.guess_type(path)
    return mt or "video/mp4"

def publish_video_direct(video_path, caption):
    """
    Tentative 'tout-en-un' (certains environnements l'acceptent) :
    POST /v2/post/publish/video/ avec fichier + caption.
    Si l'endpoint n'est pas dispo sur ton app, on bascule sur init/upload/publish.
    """
    token = ensure_access_token()
    url = f"{TIKTOK_API_BASE}/v2/post/publish/video/"
    with open(video_path, "rb") as f:
        files = {"video": (os.path.basename(video_path), f, _guess_mime(video_path))}
        data = {"caption": caption}
        out = _post(url, headers={"Authorization": f"Bearer {token}"}, data=data, files=files, timeout=120)
    return out

def publish_video_chunked(video_path, caption, chunk_size=4*1024*1024):
    """
    Flow en 3 étapes (init -> upload -> publish), plus robuste.
    """
    token = ensure_access_token()

    # 1) INIT
    init_url = f"{TIKTOK_API_BASE}/v2/post/publish/inbox/video/init/"
    file_size = os.path.getsize(video_path)
    init_body = {
        "post_info": {"caption": caption},
        "source_info": {
            "source": "VIDEO_UPLOAD",      # selon doc : VIDEO_UPLOAD ou PULL_FROM_URL
            "video_size": file_size,
            "chunk_size": chunk_size
        }
    }
    init_resp = _post(init_url, headers={"Authorization": f"Bearer {token}"}, json_body=init_body)
    upload_id = (init_resp.get("data") or {}).get("upload_id") or init_resp.get("upload_id")
    if not upload_id:
        raise RuntimeError(f"Init sans upload_id: {init_resp}")

    # 2) UPLOAD par chunks
    upload_url = f"{TIKTOK_API_BASE}/v2/post/publish/inbox/video/upload/"
    with open(video_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            files = {"video": ("chunk", chunk, _guess_mime(video_path))}
            data = {"upload_id": upload_id, "chunk_index": idx}
            _post(upload_url, headers={"Authorization": f"Bearer {token}"}, data=data, files=files, timeout=120)
            idx += 1

    # 3) PUBLISH
    publish_url = f"{TIKTOK_API_BASE}/v2/post/publish/inbox/video/complete/"
    publish_body = {"upload_id": upload_id}
    publish_resp = _post(publish_url, headers={"Authorization": f"Bearer {token}"}, json_body=publish_body)
    return publish_resp

def upload_and_publish(video_path, caption):
    """
    Essaie d'abord la voie 'directe'. Si 4xx/5xx, bascule sur le flow en 3 étapes.
    """
    try:
        return publish_video_direct(video_path, caption)
    except Exception as e:
        print("Direct publish indisponible, bascule en chunked…", e)
        return publish_video_chunked(video_path, caption)
