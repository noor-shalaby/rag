import os
import io
import json
import pickle

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


# ============================================================
# CONFIGURATION
# ===========================================================

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly"
]

CREDENTIALS_FILE = "credentials/credentials.json"

TOKEN_FILE = "token.pickle"

RAW_FOLDER_ID = "12eN15LBIXqZBxJ0ss4eijdNLtzawTAml"

OUTPUT_FOLDER = "local/raw"

MANIFEST_FILE = "local/manifest.json"


# ============================================================
# GOOGLE DRIVE AUTHENTICATION
# ============================================================

def authenticate_google_drive():

    creds = None

    # Check if we already have a saved token
    if os.path.exists(TOKEN_FILE):

        with open(TOKEN_FILE, "rb") as token:
            creds = pickle.load(token)

    # If credentials don't exist or aren't valid
    if not creds or not creds.valid:

        # Refresh expired credentials
        if creds and creds.expired and creds.refresh_token:

            creds.refresh(Request())

        else:

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES
            )

            creds = flow.run_local_server(port=0)

        # Save credentials for next time
        with open(TOKEN_FILE, "wb") as token:

            pickle.dump(creds, token)

    # Create Google Drive service
    service = build(
        "drive",
        "v3",
        credentials=creds
    )

    return service


# ============================================================
# GET FILES FROM RAW FOLDER
# ============================================================

def get_raw_files(service):

    query = (
        f"'{RAW_FOLDER_ID}' in parents "
        f"and trashed = false"
    )

    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType, size, modifiedTime)",
        pageSize=1000
    ).execute()

    files = results.get("files", [])

    return files


# ============================================================
# MANIFEST HELPERS (for incremental load)
# ============================================================

def load_manifest():

    if os.path.exists(MANIFEST_FILE):

        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def save_manifest(manifest):

    os.makedirs(
        os.path.dirname(MANIFEST_FILE),
        exist_ok=True
    )

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_files_to_download(files, manifest, output_folder):

    to_download = []

    for f in files:

        file_id = f["id"]
        modified_time = f["modifiedTime"]
        local_path = os.path.join(output_folder, f["name"])

        # New file, no record in manifest yet
        if file_id not in manifest:
            to_download.append(f)
            continue

        # Existing record but file changed on Drive
        if manifest[file_id]["modifiedTime"] != modified_time:
            to_download.append(f)
            continue

        # Record says it's downloaded, but it's missing locally
        if not os.path.exists(local_path):
            to_download.append(f)
            continue

    return to_download


# ============================================================
# DOWNLOAD ONE FILE
# ============================================================

def download_file(
    service,
    file_id,
    file_name,
    output_folder
):

    # Create output folder if it doesn't exist
    os.makedirs(
        output_folder,
        exist_ok=True
    )

    file_path = os.path.join(
        output_folder,
        file_name
    )

    print(
        f"[DOWNLOAD] {file_name}"
    )

    request = service.files().get_media(
        fileId=file_id
    )

    with open(file_path, "wb") as file:

        downloader = MediaIoBaseDownload(
            file,
            request
        )

        done = False

        while not done:

            status, done = downloader.next_chunk()

            if status:

                progress = int(
                    status.progress() * 100
                )

                print(
                    f"    Progress: {progress}%"
                )

    print(
        f"[DONE] {file_name}"
    )

    return file_path


# ============================================================
# INGESTION PIPELINE (INCREMENTAL)
# ============================================================

def ingest_data():

    print("\n==============================")
    print(" Google Drive Ingestion (Incremental)")
    print("==============================\n")

    # 1. Authenticate
    service = authenticate_google_drive()

    print("[OK] Google Drive authenticated.\n")

    # 2. Get all files currently in the RAW folder on Drive
    files = get_raw_files(service)

    print(
        f"[INFO] Found {len(files)} files in RAW folder on Drive.\n"
    )

    # 3. Load manifest of previously downloaded files
    manifest = load_manifest()

    # 4. Figure out which files are new or modified
    files_to_download = get_files_to_download(
        files,
        manifest,
        OUTPUT_FOLDER
    )

    skipped_count = len(files) - len(files_to_download)

    print(
        f"[INFO] {skipped_count} file(s) already up to date (skipped).")
    print(
        f"[INFO] {len(files_to_download)} file(s) new/modified — will download.\n"
    )

    # 5. Download only what's needed
    downloaded_files = []

    for f in files_to_download:

        file_path = download_file(
            service,
            f["id"],
            f["name"],
            OUTPUT_FOLDER
        )

        downloaded_files.append(file_path)

        # Update manifest entry right after a successful download
        manifest[f["id"]] = {
            "name": f["name"],
            "modifiedTime": f["modifiedTime"]
        }

        # Save progressively so a crash mid-run doesn't lose progress
        save_manifest(manifest)

    print("\n[OK] Ingestion finished.")
    print(f"[SUMMARY] Downloaded: {len(downloaded_files)} | Skipped: {skipped_count}\n")

    return downloaded_files


if __name__ == "__main__":
    ingest_data()
